#!/bin/bash
#
# Script should be used in a CMSSW release
#

evnum=-1

# Checking for input file

if [[ "$#" == "0" ]]; then
    echo "usage: './runOffline.sh inputfile' ";
    exit 1;
fi

eval `scramv1 r -sh`

# creating the configuration file to process raw data

cmsDriver.py recoDQM -s RAW2DIGI,RECO,DQM -n ${evnum} --eventcontent DQM --conditions auto:com10 --geometry Ideal --filein $1 --data --no_exec --python_filename=recoDQM_ref.py

cmsRun -e recoDQM_ref.py >& recoDQM.log 

# creating the configuration file to process DQM output file

cmsDriver.py offlineDQM -s HARVESTING:dqmHarvesting --conditions auto:com10 --data --filein file:recoDQM_RAW2DIGI_RECO_DQM.root --scenario pp  --no_exec --python_filename=offlineDQM_ref.py


cmsRun -e offlineDQM_ref.py >& offlineDQM.log 