#!/bin/sh
echo " "
echo "Sending job to LSF (batch system) ..."
#job submission
#using "pool" options the output file will be written
#under /pool/lsf/user_name/job_id

for (( i = 1;  i <= nFiles; ++i ))
    do
    cp py_config/run_job_LAS_RunXXXXXX_EvYYYYYY_F$i.sh .
    bsub -R "pool>10000" -q 1nh run_job_LAS_RunXXXXXX_EvYYYYYY_F$i.sh
    done
echo "Please check with \"bjobs\" the status!"
