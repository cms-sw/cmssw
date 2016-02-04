#! /bin/env python

# Rel val launcher

import os
import sys
import time


if len(sys.argv)!=6:
    print "Launcher:"
    print "Usage: "+sys.argv[0]+" step numberOfEvents workdir queue cmssw_base\n\n"
    raise "Too Few Parameters."

step=sys.argv[1]
nevts=sys.argv[2]
my_workdir=sys.argv[3]
queue=sys.argv[4]
cmssw_base=sys.argv[5]

qed_ene="10"
jet_en="50_120"
type_energy_dict={"MU+":qed_ene,
                  "MU-":qed_ene,
                  "E+":qed_ene,
                  "E-":qed_ene,
                  "GAMMA":qed_ene,
                  #"10MU+":qed_ene,
                  "10MU-":qed_ene,
                  #"10E+":qed_ene,
                  #"10E-":qed_ene,
                  #"10GAMMA":qed_ene,
                  "QCD":"380_470",
                  "B_JETS":jet_en,"C_JETS":jet_en,"UDS_JETS":jet_en,
                  "ZPJJ":"",
                  "HZZEEEE":"","HZZMUMUMUMU":"",
                  "TTBAR":"",
                  "BSJPSIPHI":"",
                  "TAU":"20_420"}
                  
#state the location of cmsDriver

executable="$CMSSW_BASE/src/Configuration/PyReleaseValidation/data/cmsDriver.py"  

for evt_type in type_energy_dict.keys():                             
    job_content=\
"""#! /bin/sh
cd """+cmssw_base+"""
eval `scramv1 runtime -sh`
PYTHONPATH=$PYTHONPATH:$CMSSW_BASE/src/Configuration/PyReleaseValidation/data
cd """+my_workdir+"""/"""+evt_type+"""
"""+executable+""" """+evt_type+""" -s """+step+""" -n """+nevts
    
    if not os.path.exists(evt_type):
        os.mkdir(evt_type)
    job_name=my_workdir+"/"+evt_type+"/test_job_"+evt_type+".sh"
    job=file(job_name,"w")
    job.write(job_content)
    job.close()
    os.system("chmod +x "+job_name)
    os.system ("bsub -q "+queue+" "+job_name)            
