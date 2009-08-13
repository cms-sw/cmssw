#!/bin/sh

for (( i = 1;  i <= nFiles; ++i ))
    do
        cp run_job_LAS_RunXXXXXX_EvYYYYYY.sh run_job_LAS_RunXXXXXX_EvYYYYYY_F$i.sh
	sed -i "s/_F0/_F$i/g" run_job_LAS_RunXXXXXX_EvYYYYYY_F$i.sh
	mv run_job_LAS_RunXXXXXX_EvYYYYYY_F$i.sh py_config
    done

