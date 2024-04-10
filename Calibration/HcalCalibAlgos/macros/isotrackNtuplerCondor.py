from pathlib import Path
import json
import os
from optparse import OptionParser
import subprocess
import uproot3

parser=OptionParser()
MYDIR=os.getcwd()

parser.add_option("-q","--flavour",dest="jobFlavour",type="str",default="workday",help="job FLAVOUR",metavar="FLAVOUR")

parser.add_option("--filePU",dest="PU_file",type=str, default=None,help="input PU file")

parser.add_option("--isotrackNtupler", dest="IsotrackNtupler", type=str, default=None, help="location of isotrackNtupler")




parser.add_option("--fileNPU", dest="NoPU_file", type=str, default=None,
                    help="input NoPU file")

parser.add_option("--output", "-o", dest="output_dir", type=str,
                    help="location of output file directory without '/' ", )

opts, args = parser.parse_args()

jobDir = MYDIR + '/condor_jobs_vF4'
Path(jobDir).mkdir(parents=True, exist_ok=False)
job_files = []

PU_tree = uproot3.open(opts.PU_file)['hcalIsoTrkAnalyzer/CalibTree']
pu_entries = PU_tree.numentries
print("No of elements in pileup samples",pu_entries)

scale = 5000000
pu_start = 0
i = 0

print("Creating jobs ------->")
for index in range(0,pu_entries, scale):
    print(i)
    pu_stop = index+scale
    if (pu_stop > pu_entries):
        pu_stop = pu_entries
    output_file = opts.output_dir + "/IsoTrkNtupler"
    arguments = "%s -NPU %s -PU %s --start %i --end %i -O %s"%(opts.IsotrackNtupler, opts.NoPU_file, opts.PU_file, pu_start, pu_stop, output_file)
    pu_start = pu_stop
    file_name = "IsoTrkNtupler_" + str(i)
    job_file_name = os.path.join(jobDir, f"{file_name}.sub")
    job_file_out = os.path.join(jobDir, f"{file_name}.out")
    job_file_err = os.path.join(jobDir, f"{file_name}.err")

    with open(job_file_name, "w") as submit_file:
        submit_file.write(f"executable = /eos/user/d/dasgupsu/anaconda3/envs/dna/bin/python3.8\n")
        submit_file.write(f"arguments =  {arguments}\n")
        submit_file.write(f"output = {job_file_out}\n")
        submit_file.write(f"error = {job_file_err}\n")
        submit_file.write("getenv = True\n")
        submit_file.write(f'+JobFlavour = "{opts.jobFlavour}"\n')
        submit_file.write("queue 1\n")
    job_files.append(job_file_name)
    i+= 1

for jf in job_files:
    if jobDir.startswith("/eos"):
        subprocess.run(["condor_submit", "-spool", jf])
    else:                
        subprocess.run(["condor_submit", jf])
