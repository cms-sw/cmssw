#!/usr/bin/env python3
#test execute: export CMSSW_BASE=/tmp/CMSSW && ./validateAlignments.py -c defaultCRAFTValidation.ini,test.ini -n -N test
from __future__ import print_function
from future.utils import lmap
import subprocess
import json
import yaml
import os
import argparse
import pprint
import sys
import shutil
import Alignment.OfflineValidation.TkAlAllInOneTool.findAndChange as fnc

import Alignment.OfflineValidation.TkAlAllInOneTool.GCP as GCP
import Alignment.OfflineValidation.TkAlAllInOneTool.DMR as DMR
import Alignment.OfflineValidation.TkAlAllInOneTool.Zmumu as Zmumu
import Alignment.OfflineValidation.TkAlAllInOneTool.PV as PV
import Alignment.OfflineValidation.TkAlAllInOneTool.SplitV as SplitV
import Alignment.OfflineValidation.TkAlAllInOneTool.JetHT as JetHT

##############################################
def parser():
##############################################
    """ Parse user input """

    parser = argparse.ArgumentParser(description = "AllInOneTool for validation of the tracker alignment", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", metavar='config', type=str, action="store", help="Global AllInOneTool config (json/yaml format)")
    parser.add_argument("-d", "--dry", action = "store_true", help ="Set up everything, but don't run anything")
    parser.add_argument("-v", "--verbose", action = "store_true", help ="Enable standard output stream")
    parser.add_argument("-e", "--example", action = "store_true", help ="Print example of config in JSON format")
    parser.add_argument("-f", "--force", action = "store_true", help ="Force creation of enviroment, possible overwritten old configuration")
    parser.add_argument("-j", "--job-flavour", action = "store", default = "workday", choices = ["espresso", "microcentury", "longlunch", "workday", "tomorrow", "testmatch", "nextweek"], help ="Job flavours for HTCondor at CERN, default is 'workday'")

    return parser.parse_args()

##############################################
def check_proxy():
##############################################
    """Check if GRID proxy has been initialized."""

    try:
        with open(os.devnull, "w") as dump:
            subprocess.check_call(["voms-proxy-info", "--exists"],
                                  stdout = dump, stderr = dump)
    except subprocess.CalledProcessError:
        return False
    return True

##############################################
def forward_proxy(rundir):
##############################################
    """Forward proxy to location visible from the batch system.
    Arguments:
    - `rundir`: directory for storing the forwarded proxy
    Return:
    - Full path to the forwarded proxy
    """

    if not check_proxy():
        print("Please create proxy via 'voms-proxy-init -voms cms'.")
        sys.exit(1)

    ## Move the proxy to the run directory
    proxyName = "{}/.user_proxy".format(rundir)
    localProxy = subprocess.check_output(["voms-proxy-info", "--path"]).strip()
    shutil.copyfile(localProxy, proxyName)

    ## Return the path to the forwarded proxy
    return proxyName


##############################################
def updateConfigurationFile(configurationFile, updateInstructions):
##############################################
    """Update a template configuration file with custom configuration
    Arguments:
    - configurationFile: File name for the configuration file that will be updated
    - updateInstructions: A dictionary defining the updated configuration with keys "overwrite", "remove", "add" and "addBefore" each containing a list with the instructions on what should be replaced, removed or added.
    """

    # Read the original configuration file
    with open(configurationFile,"r") as inputFile:
        fileContent = inputFile.readlines()

    # Perform all overwrite operations to the configuration file. First string where the substring before the first space matches with the replacing string is overwritten. If a character "|" is included in the instruction, the subtring before that is used to search for the overwritten line instead. If no such string is found, add the instruction to the end of the file.
    if "overwrite" in updateInstructions:

        for instruction in updateInstructions["overwrite"]:

            decodeInstruction = instruction.split("|")
            if(len(decodeInstruction) > 1):
                lineToReplace = decodeInstruction[0]
                newInstruction = instruction[instruction.index("|")+1:]
            else:
                lineToReplace = instruction.split()[0]
                newInstruction = instruction

            lineOverwritten = False
            for iLine in range(0,len(fileContent)):
                if fileContent[iLine].startswith(lineToReplace):
                    fileContent[iLine] = newInstruction
                    if not fileContent[iLine].endswith("\n"):
                        fileContent[iLine] = fileContent[iLine] + "\n"
                    lineOverwritten = True
                    break

            # If did not find a line to overwrite, add the instruction to the end of the file
            if not lineOverwritten:
                fileContent.append(newInstruction)
                if not fileContent[-1].endswith("\n"):
                    fileContent[-1] = fileContent[-1] + "\n"

    # Perform all remove operations to the configuration file. First string that starst with the instruction will be removed from the configuration file.
    if "remove" in updateInstructions:
        for instruction in updateInstructions["remove"]:
            for iLine in range(0,len(fileContent)):
                if fileContent[iLine].startswith(instruction):
                    fileContent.pop(iLine)
                    break

    # Perform all add operations to the configuration file. The instruction is added to the matching CRAB configuration section. If one is not found, it is added to the end of the file.
    if "add" in updateInstructions:
        for instruction in updateInstructions["add"]:
            categories = instruction.split(".")
            if len(categories) > 2:
                category = categories[1]
            else:
                category = "nonExistent"
            previousCategory = ""
            lineFound = False

            # First try to add the line to a correct section in CRAB configuration
            for iLine in range(0,len(fileContent)):
                if fileContent[iLine] == "\n" and previousCategory == category:
                    fileContent.insert(iLine, instruction)
                    if not fileContent[iLine].endswith("\n"):
                        fileContent[iLine] = fileContent[iLine] + "\n"
                    lineFound = True
                    break
                elif fileContent[iLine] == "\n":
                    previousCategory = ""
                else:
                    newCategories = fileContent[iLine].split(".")
                    if len(newCategories) > 2:
                        previousCategory = newCategories[1]
                    else:
                        previousCategory = ""

            # If the correct section is not found, add the new line to the end of the file
            if not lineFound:
                fileContent.append(instruction)
                if not fileContent[-1].endswith("\n"):
                    fileContent[-1] = fileContent[-1] + "\n"

    # Perform all addBefore operations to the configuration file. This adds an instruction to the configuration file just before a line that starts with a string defined before the '|' character. If one is not found, the line is added to the end of the file.
    if "addBefore" in updateInstructions:
        for instruction in updateInstructions["addBefore"]:
            lineBefore = instruction.split("|")[0]
            newInstruction = instruction[instruction.index("|")+1:]
            lineFound = False
            for iLine in range(0,len(fileContent)):
                if fileContent[iLine].startswith(lineBefore):
                    fileContent.insert(iLine,newInstruction)
                    if not fileContent[iLine].endswith("\n"):
                        fileContent[iLine] = fileContent[iLine] + "\n"
                    lineFound = True
                    break


            # If the searched line is not found, add the new line to the end of the file
            if not lineFound:
                fileContent.append(newInstruction)
                if not fileContent[-1].endswith("\n"):
                    fileContent[-1] = fileContent[-1] + "\n"

    # Write the updates to the configuration file
    with open(configurationFile,"w") as outputFile:
        outputFile.writelines(fileContent)


##############################################
def main():
##############################################

    ## Before doing anything, check that grip proxy exists
    if not check_proxy():
        print("Grid proxy is required in most use cases of the tool.")
        print("Please create a proxy via 'voms-proxy-init -voms cms'.")
        sys.exit(1)

    ##Read parser arguments
    args = parser()

    ##Print example config which is in Aligment/OfflineValidation/bin if wished
    if args.example:
        with open("{}/src/Alignment/OfflineValidation/bin/example.yaml".format(os.environ["CMSSW_BASE"]), "r") as exampleFile:
            config = yaml.load(exampleFile, Loader=yaml.Loader)
            pprint.pprint(config, width=30)
            sys.exit(0)    

    ##Read in AllInOne config dependent on what format you choose
    with open(args.config, "r") as configFile:
        if args.verbose:
            print("Read AllInOne config: '{}'".format(args.config))

        if args.config.split(".")[-1] == "json":
            config = json.load(configFile)

        elif args.config.split(".")[-1] == "yaml":
            config = yaml.load(configFile, Loader=yaml.Loader)

        else:
            raise Exception("Unknown config extension '{}'. Please use json/yaml format!".format(args.config.split(".")[-1])) 

    ##Check for all paths in configuration and attempt to "digest" them
    for path in fnc.find_and_change(list(), config):
        if args.verbose and ("." in str(path) or "/" in str(path)):
            print("Digesting path: "+str(path))
         
    ##Create working directory
    if os.path.isdir(config["name"]) and not args.force:
        raise Exception("Validation directory '{}' already exists! Please choose another name for your directory.".format(config["name"]))	

    validationDir = os.path.abspath(config["name"])
    exeDir = "{}/executables".format(validationDir)
    cmsconfigDir =  "{}/cmsConfigs".format(validationDir)

    subprocess.call(["mkdir", "-p", validationDir] + ((["-v"] if args.verbose else [])))
    subprocess.call(["mkdir", "-p", exeDir] + (["-v"] if args.verbose else []))
    subprocess.call(["mkdir", "-p", cmsconfigDir] + (["-v"] if args.verbose else []))

    ##Copy AllInOne config in working directory in json/yaml format
    subprocess.call(["cp", "-f", args.config, validationDir] + (["-v"] if args.verbose else []))

    ## Define the template files
    crabTemplateFile = fnc.digest_path("$CMSSW_BASE/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/templates/crabTemplate.py")    
    condorTemplateFile = fnc.digest_path("$CMSSW_BASE/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/templates/condorTemplate.submit")
    executableTempleteFile = fnc.digest_path("$CMSSW_BASE/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/templates/executableTemplate.sh")
    

    ##List with all jobs
    jobs = []

    ##Check in config for all validation and create jobs
    for validation in config["validations"]:
        if validation == "GCP":
            jobs.extend(GCP.GCP(config, validationDir))

        elif validation == "DMR":
            jobs.extend(DMR.DMR(config, validationDir))

        elif validation == "Zmumu":
            jobs.extend(Zmumu.Zmumu(config, validationDir))

        elif validation == "PV":
            jobs.extend(PV.PV(config, validationDir))

        elif validation == "SplitV":
            jobs.extend(SplitV.SplitV(config, validationDir))

        elif validation == "JetHT":
            jobs.extend(JetHT.JetHT(config, validationDir))

        else:
            raise Exception("Unknown validation method: {}".format(validation)) 
            
    ##Create dir for DAG file and loop over all jobs
    subprocess.call(["mkdir", "-p", "{}/DAG/".format(validationDir)] + (["-v"] if args.verbose else []))

    with open("{}/DAG/dagFile".format(validationDir), "w") as dag:
        for job in jobs:
            ##Create job dir, output dir
            subprocess.call(["mkdir", "-p", job["dir"]] + (["-v"] if args.verbose else []))
            subprocess.call(["mkdir", "-p", job["config"]["output"]] + (["-v"] if args.verbose else []))
            subprocess.call(["mkdir", "-p", "{}/condor".format(job["dir"])] + (["-v"] if args.verbose else []))
            subprocess.call(["ln", "-fs", job["config"]["output"], "{}/output".format(job["dir"])] + (["-v"] if args.verbose else []))
            
            ## Copy the template files to the job directory
            crabConfigurationFile = "{}/crabConfiguration.py".format(job["dir"])
            subprocess.call(["cp", crabTemplateFile, crabConfigurationFile] + (["-v"] if args.verbose else []))
            condorSubmitFile = "{}/condor.sub".format(job["dir"])
            subprocess.call(["cp", condorTemplateFile, condorSubmitFile] + (["-v"] if args.verbose else []))
            executableFile = "{}/run.sh".format(job["dir"])
            subprocess.call(["cp", executableTempleteFile, executableFile] + (["-v"] if args.verbose else []))

            ## Forward the proxy to the job directory
            if args.verbose:
                print("Forwarding grid proxy to directory {}".format(job["dir"]))
            myProxy = forward_proxy(job["dir"])

            ##Create symlink for executable/python cms config if needed
            subprocess.call("cp -f $(which {}) {}".format(job["exe"], exeDir) + (" -v" if args.verbose else ""), shell = True)
            subprocess.call(["ln", "-fs", "{}/{}".format(exeDir, job["exe"]), job["dir"]] + (["-v"] if args.verbose else []))
            if "cms-config" in job:
                cmsConfig = job["cms-config"].split("/")[-1]

                subprocess.call(["cp", "-f", job["cms-config"], "{}/{}".format(cmsconfigDir, cmsConfig)] + (["-v"] if args.verbose else []))
                subprocess.call(["ln", "-fs", "{}/{}".format(cmsconfigDir, cmsConfig), "{}/validation_cfg.py".format(job["dir"])] + (["-v"] if args.verbose else []))

            ##Write local config file 
            with open("{}/validation.json".format(job["dir"]), "w") as jsonFile:
                if args.verbose:
                    print("Write local json config: '{}'".format("{}/validation.json".format(job["dir"])))           

                json.dump(job["config"], jsonFile, indent=4)

            ## Customize the executable template file for this specific job
            executableCustomization = {"overwrite": [], "addBefore": []}

            executableCustomization["overwrite"].append("export X509|export X509_USER_PROXY={}".format(myProxy)) # Define the proxy location
            executableCustomization["overwrite"].append("cd workDir|cd {}".format(job["dir"])) # Define the work directory for this job

            # Option the give free arguments to the executable
            if "exeArguments" in job:
                executableCustomization["overwrite"].append("./cmsRun|./{} {}".format(job["exe"], job["exeArguments"])) # Define the correct executable for this job
            else: # Default arguments
                executableCustomization["overwrite"].append("./cmsRun|./{} {}validation.json".format(job["exe"], "validation_cfg.py config=" if "cms-config" in job else "")) # Define the correct executable for this job

            # Option to include the condor job number given as a command line argument
            if "nCondorJobs" in job:
                executableCustomization["addBefore"].append("./{}|JOBNUMBER=${{1:--1}}".format(job["exe"]))

            # Do the manual configuration on top of the executable file
            updateConfigurationFile(executableFile, executableCustomization)

            # Give the correct access rights for the executable
            subprocess.call(["chmod", "a+rx", executableFile] + (["-v"] if args.verbose else []))

            ## Customize the condor submit file for this specific job
            condorSubmitCustomization = {"overwrite": [], "addBefore": []}

            # Take given flavour for the job, except if overwritten in job config
            condorSubmitCustomization["overwrite"].append('+JobFlavour = "{}"'.format(args.job_flavour if not 'flavour' in job else job['flavour']))
            
            # If condor job array is sent, add job ID information to submit file
            if "nCondorJobs" in job:
                condorSubmitCustomization["addBefore"].append("output|arguments = $(ProcID)")
                condorSubmitCustomization["overwrite"].append("output = condor/condor$(ProcID).out")
                condorSubmitCustomization["overwrite"].append("error  = condor/condor$(ProcID).err")
                condorSubmitCustomization["overwrite"].append("log    = condor/condor$(ProcID).log")
                condorSubmitCustomization["overwrite"].append("queue {}".format(job["nCondorJobs"]))

            # Do the customization for the condor submit file
            updateConfigurationFile(condorSubmitFile, condorSubmitCustomization)

            ##Write command in dag file
            dag.write("JOB {} condor.sub DIR {}\n".format(job["name"], job["dir"]))

            if job["dependencies"]:
                dag.write("\n")
                dag.write("PARENT {} CHILD {}".format(" ".join(job["dependencies"]), job["name"]))

            dag.write("\n\n")

            ## If there is custom crab configuration defined, modify the crab template file based on that
            if "crabCustomConfiguration" in job["config"]:
                updateConfigurationFile(crabConfigurationFile, job["config"]["crabCustomConfiguration"])


    if args.verbose:
        print("DAGman config has been written: '{}'".format("{}/DAG/dagFile".format(validationDir)))            

    ##Call submit command if not dry run
    if args.dry:
        print("Enviroment is set up. If you want to submit everything, call 'condor_submit_dag {}/DAG/dagFile'".format(validationDir))

    else:
        subprocess.call(["condor_submit_dag", "{}/DAG/dagFile".format(validationDir)])
        
##############################################
if __name__ == "__main__":
##############################################
    main()
