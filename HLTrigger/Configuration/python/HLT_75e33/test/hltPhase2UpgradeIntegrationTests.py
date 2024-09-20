import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)

# Automatically generate the default geometry from DEFAULT_VERSION
_PH2_GEOMETRY = f"Extended{_settings.DEFAULT_VERSION}"

# Get the actual era name from the version key
_PH2_ERA_NAME = _settings.properties[2026][_settings.DEFAULT_VERSION]['Era']

# Function to display help information
def print_help():
    help_text = """
    This script runs HLT test configurations for the CMS Phase2 upgrade.

    Arguments:
    --globaltag    : GlobalTag for the CMS conditions (required)
    --geometry     : Geometry setting for the CMS process (required)
    --events       : Number of events to process (default: 1)
    --threads      : Number of threads to use (default: 1)

    Example usage:
    python script.py --globaltag auto:phase2_realistic_T33 --geometry Extended2026D110 --events 10 --threads 4
    """
    print(help_text)

# Function to run a shell command and handle errors
def run_command(command, log_file=None, workdir=None):
    try:
        print(f"Running command: {command}")
        with open(log_file, "w") as log:
            subprocess.run(command, shell=True, check=True, cwd=workdir, stdout=log, stderr=log)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return e.returncode  # Return the error code
       
    return 0  # Return 0 for success
       
# Argument Parser for command-line configuration
parser = argparse.ArgumentParser(description="Run HLT Test Configurations")
parser.add_argument("--globaltag", default=_PH2_GLOBAL_TAG, help="GlobalTag for the CMS conditions")
parser.add_argument("--geometry", default=_PH2_GEOMETRY, help="Geometry setting for the CMS process")  # Auto-generated geometry default
parser.add_argument("--era", default=_PH2_ERA_NAME, help="Era setting for the CMS process")  # Convert _PH2_ERA to string
parser.add_argument("--events", type=int, default=10, help="Number of events to process")
parser.add_argument("--parallelJobs", type=int, default=4, help="Number of parallel cmsRun HLT jobs")
parser.add_argument("--threads", type=int, default=1, help="Number of threads to use")

# Step 0: Capture the start time and print the start timestamp
start_time = time.time()
start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("------------------------------------------------------------")
print(f"Script started at {start_timestamp}")
print("------------------------------------------------------------")

# Parse arguments
try:
    args = parser.parse_args()
except SystemExit:
    print_help()
    sys.exit(0)

global_tag = args.globaltag
era = args.era
geometry = args.geometry
num_events = args.events
num_threads = args.threads
num_parallel_jobs = args.parallelJobs

# Print the values in a nice formatted manner
print(f"{'Configuration Summary':^40}")
print("=" * 40)
print(f"Global Tag:    {global_tag}")
print(f"Geometry:      {geometry}")
print(f"Era:           {era}")
print(f"Num Events:    {num_events}")
print(f"Num Threads:   {num_threads}")
print(f"Num Par. Jobs: {num_parallel_jobs}")
print("=" * 40)

# Directory where all test configurations will be stored
output_dir = "hlt_test_configs"
# If the directory exists, remove it first
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create the directory
os.makedirs(output_dir)

# Define the cmsDriver.py command to create the base configuration
base_cmsdriver_command = (
    f"cmsDriver.py Phase2 -s L1P2GT,HLT:75e33_timing "
    f"--conditions {global_tag} -n {num_events} --eventcontent FEVTDEBUGHLT "
    f"--geometry {geometry} --era {era} --filein file:{output_dir}/step1.root --fileout {output_dir}/hlt.root --no_exec "
    f"--nThreads {num_threads} "
    f"--customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000 "
    f'--customise_commands "process.options.wantSummary=True"'
)

# The base configuration file and the dumped configuration file
base_config_file = os.path.join(output_dir, "Phase2_L1P2GT_HLT.py")
dumped_config_file = os.path.join(output_dir, "Phase2_dump.py")
log_file = os.path.join(output_dir, "hlt.log")

# Step 1: Run the cmsDriver.py command to generate the base configuration in the output directory
print(f"Running cmsDriver.py to generate the base config: {base_config_file}")
subprocess.run(base_cmsdriver_command, shell=True, cwd=output_dir)

# Step 2: Use edmConfigDump to dump the full configuration
print(f"Dumping the full configuration using edmConfigDump to {dumped_config_file}")
with open(dumped_config_file, "w") as dump_file, open(log_file, "w") as log:
    subprocess.run(f"edmConfigDump {base_config_file}", shell=True, stdout=dump_file, stderr=log)

# Step 3: Extract the list of HLT paths from the dumped configuration
print(f"Extracting HLT paths from {dumped_config_file}...")

# Read the dumped configuration to extract HLT paths
with open(dumped_config_file, "r") as f:
    config_content = f.read()

# Use regex to find all HLT and L1T paths defined in process.schedule
unsorted_hlt_paths = re.findall(r"process\.(HLT_[A-Za-z0-9_]+|L1T_[A-Za-z0-9_]+)", config_content)

# Remove duplicates and sort alphabetically
hlt_paths = sorted(set(unsorted_hlt_paths))

if not hlt_paths:
    print("No HLT paths found in the schedule!")
    exit(1)

print(f"Found {len(hlt_paths)} HLT paths.")

# Step 4: Broadened Regex for Matching process.schedule
schedule_match = re.search(
    r"(process\.schedule\s*=\s*cms\.Schedule\(\*?\s*\[)([\s\S]+?)(\]\s*\))", 
    config_content
)

if not schedule_match:
    print("No schedule match found after tweaking regex! Exiting...")
    exit(1)
else:
    print(f"Matched schedule section.")

# Step 5: Generate N configurations by modifying the dumped config to keep only one path at a time
for path_name in hlt_paths:
    # Create a new configuration file for this path
    config_filename = os.path.join(output_dir, f"Phase2_{path_name}.py")
    
    # Define regex to find all HLT paths in the cms.Schedule and replace them
    def replace_hlt_paths(match):
        all_paths = match.group(2).split(", ")
        # Keep non-HLT/L1T paths and include only the current HLT or L1T path
        filtered_paths = [path for path in all_paths if not re.match(r"process\.(HLT_|L1T_)", path) or f"process.{path_name}" in path]
        return match.group(1) + ", ".join(filtered_paths) + match.group(3)

    # Apply the regex to remove all HLT and L1T paths except the current one
    modified_content = re.sub(
        r"(process\.schedule\s*=\s*cms\.Schedule\(\*?\s*\[)([\s\S]+?)(\]\s*\))",
        replace_hlt_paths,
        config_content
    )

    # Modify the fileout parameter to save a unique root file for each path
    modified_content = re.sub(
        r"fileName = cms\.untracked\.string\('.*'\)", 
        f"fileName = cms.untracked.string('{output_dir}/{path_name}.root')", 
        modified_content
    )

    # Write the new config to a file
    with open(config_filename, "w") as new_config:
        new_config.write(modified_content)
    
    print(f"Generated config: {config_filename}")

print(f"Generated {len(hlt_paths)} configuration files in the {output_dir} directory.")

# Step 6: Run cmsDriver.py for TTbar GEN,SIM steps and save the output in output_dir
ttbar_config_file = os.path.join(output_dir, "TTbar_GEN_SIM_step.py")
ttbar_command = (
    f"cmsDriver.py TTbar_14TeV_TuneCP5_cfi -s GEN,SIM,DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW -n {num_events} "
    f"--conditions {global_tag} --beamspot DBrealisticHLLHC --datatier GEN-SIM-DIGI-RAW "
    f"--eventcontent FEVTDEBUG --geometry {geometry} --era {era} "
    f"--relval 9000,100 --fileout {output_dir}/step1.root --nThreads {num_threads} "
    f"--customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000 "
    f"--python_filename {ttbar_config_file}"
)

print("Running TTbar GEN,SIM step...")
run_command(ttbar_command, log_file=os.path.join(output_dir, "ttbar_gen_sim.log"))

# Directory containing HLT test configurations
hlt_configs_dir = output_dir

# Check if the directory exists
if not os.path.exists(hlt_configs_dir):
    print(f"Directory {hlt_configs_dir} not found! Exiting...")
    exit(1)

# Step 7: Function to run cmsRun on a given HLT config file and save the output
def run_cmsrun(config_file):
    # Extract the HLT path name from the config file (e.g., "Phase2_HLT_IsoMu24_FromL1TkMuon.py")
    base_name = os.path.basename(config_file).replace("Phase2_", "").replace(".py", "")
    log_file = os.path.join(output_dir, f"{base_name}.log")
    
    # Run the cmsRun command and log the output
    cmsrun_command = f"cmsRun {config_file}"
    result_code = run_command(cmsrun_command, log_file=log_file)

    if result_code != 0:
        print(f"cmsRun failed for {config_file} with exit code {result_code}. Check {log_file} for details.")
        return result_code  # Return the error code

    print(f"cmsRun completed for {config_file}")
    return 0  # Return 0 for success
    
# Step 8: Loop through all files in hlt_test_configs and run cmsRun on each in parallel
config_files = [f for f in os.listdir(hlt_configs_dir) if f.endswith(".py") and f.startswith("Phase2_")]
print(f"Found {len(config_files)} configuration files in {hlt_configs_dir}.")

# Run cmsRun on all config files in parallel and handle errors
error_occurred = False
with ThreadPoolExecutor(max_workers=num_parallel_jobs) as executor:
    futures = {executor.submit(run_cmsrun, os.path.join(output_dir, config_file)): config_file for config_file in config_files}

    for future in as_completed(futures):
        config_file = futures[future]
        try:
            result_code = future.result()
            if result_code != 0:
                error_occurred = True
                print(f"cmsRun for {config_file} exited with code {result_code}")
        except Exception as exc:
            error_occurred = True
            print(f"cmsRun for {config_file} generated an exception: {exc}")

if error_occurred:
    print("One or more cmsRun jobs failed. Exiting with failure.")
    exit(1)

print("All cmsRun jobs submitted.")

# Step 9: Compare all HLT root files using hltDiff
def compare_hlt_results():
    # List all root files starting with "HLT_" in the output directory
    root_files = [f for f in os.listdir(output_dir) if f.endswith(".root") and (f.startswith("HLT_") or f.startswith("L1T_"))]
    
    # Base file (hltrun output) to compare against
    base_root_file = os.path.join(output_dir, "hlt.root")
    
    # Check if base_root_file exists
    if not os.path.exists(base_root_file):
        print(f"Base root file {base_root_file} not found! Exiting...")
        exit(1)

    # Compare each root file with the base using hltDiff
    for root_file in root_files:
        root_path = os.path.join(output_dir, root_file)
        print(f"Comparing {root_path} with {base_root_file} using hltDiff...")
        
        # Run the hltDiff command
        hlt_diff_command = f"hltDiff -o {base_root_file} -n {root_path}"
        result = subprocess.run(hlt_diff_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Decode and process the output
        output = result.stdout.decode("utf-8")
        print(output)  # Print for debug purposes

        # Use a dynamic check based on the number of events configured
        expected_match_string = f"Found {num_events} matching events, out of which 0 have different HLT results"

        # Check if the output contains the expected match string
        if expected_match_string not in output:
            print(f"Error: {root_file} has different HLT results!")
            exit(1)  # Exit with failure if differences are found

    print("All HLT comparisons passed with no differences.")

# Step 10: Once all cmsRun jobs are completed, perform the hltDiff comparisons
print("Performing HLT result comparisons...")
compare_hlt_results()

# Step 11: Capture the end time and print the end timestamp
end_time = time.time()
end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("------------------------------------------------------------")
print(f"Script ended at {end_timestamp}")
print("------------------------------------------------------------")

# Step 12: Calculate the total execution time and print it
total_time = end_time - start_time
formatted_total_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
print("------------------------------------------------------------")
print(f"Total execution time: {formatted_total_time}")
print("------------------------------------------------------------")

print("All steps completed successfully.")
