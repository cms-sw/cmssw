#! /bin/bash

#to use:
#set the input geometry and IOV in testProduceSystematicMisalignment_cfg.py
#set the output file name to outputfile.db or folder/outputfile.db
#  (outputfile.db is replaced by (misalignment name).db)

#set the magnitude for each one in the comment after the -999
#set the phases for the phi dependent misalignments

#then cmsenv and run this
#don't bother submitting to a queue, the whole thing takes less than 2 minutes

folder=$CMSSW_BASE/src/Alignment/TrackerAlignment/test

for a in $(grep Epsilon $folder/testProduceSystematicMisalignment_cfg.py  | grep 999 | sed "s/Epsilon.*//"); do
    sed -r -e "s/(${a}Epsilon.*)-999. *#/\1/" -e "s|outputfile.db|${a}.db|" $folder/testProduceSystematicMisalignment_cfg.py > ${a}_cfg.py
    cmsRun ${a}_cfg.py
    rm ${a}_cfg.py
done
