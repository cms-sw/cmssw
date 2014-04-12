#!/bin/bash
#
# This script is part of the Kalman Alignment Production System (KAPS).
#
# This script contains some useful functions to ease the steering of
# consecutive alignment jobs using KAPS. It has to be run in the
# working directory of the alignment jobs, i.e. the directory that
# contains the template scripts, template cfg-files, etc.

function jobs_still_running
{
    RESULT_RUN=`kaps_stat.pl | grep -C 0 RUN`
    RESULT_PEND=`kaps_stat.pl | grep -C 0 PEND`
    if [[ -z "$RESULT_RUN" && -z "$RESULT_PEND" ]]; then
	echo "[job_still_runnning] return 0"
	return 0
    else
       	echo "[job_still_runnning] return 1"
	return 1
    fi
}

function monitor_jobs
{
    stat=1
    while [ "$stat" -eq 1 ]; do
	sleep $1
	jobs_still_running
	stat=$?
    done
}

function clean_up
{
    rm -r jobData
    rm -r LSFJOB_*
    rm -r kaps.db
}

function alignment_run
{
    echo "[alignment_run] using the following parameters"
    echo "  template shell script = " $1
    echo "  template config = " $2
    echo "  data sample = " $3
    echo "  number of jobs = " $4
    echo "  job descriptor = " $5
    echo "  template merge shell script" $6
    echo "  storage directory = " $7
    kaps_setup.pl -m $1 $2 $3 $4 cmscaf $5 $6 $7
    #kaps_setup.pl -m $1 $2 $3 $4 8nm $5 $6 $7
    #kaps_fire.pl $4
    #monitor_jobs 30
    #kaps_fetch.pl
    #kaps_fire.pl -mf
    #monitor_jobs 30
    #clean_up
}

function fetch_output
{
    rm $1/kaaOutput*.root
    rename kaaMerged $2 $1/kaaMerged.root
}
