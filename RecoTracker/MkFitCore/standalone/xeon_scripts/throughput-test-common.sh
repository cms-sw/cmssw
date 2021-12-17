#!/bin/bash

source xeon_scripts/stress-test-common.sh

####################
## Core Test Loop ##
####################

function MkFitThroughput ()
{
    local test_exe=${1}
    local nproc=${2}
    local njob=${3}
    local tmp_output_file=${4}

    ## launch jobs in parallel to background : let scheduler put jobs all around
    for (( ijob = 0 ; ijob < ${njob} ; ijob++ ))
    do
        ## run the executable
	{ time ${test_exe} --num-events ${nproc} > /dev/null 2>&1 ; } 2> "${tmp_output_file}.${ijob}" &
    done

    ## wait for all background processes to finish --> non-ideal as we would rather "stream" jobs launching
    wait
}
export -f MkFitThroughput

####################################
## Dump Tmp Output into Main File ##
####################################

function DumpIntoFileThroughput ()
{
    local tmp_output_file=${1}
    local output_file=${2}

    ## get wall-clock time, split
    total_time=0
    while read -ra time_arr
    do
        local tmp_time=${time_arr[1]}
        local mins=$( echo "${tmp_time}" | cut -d "m" -f 1 )
        local secs=$( echo "${tmp_time}" | cut -d "m" -f 2  | cut -d "s" -f 1 )
        local total_time=$( bc -l <<< "${total_time} + ${mins} * 60 + ${secs}" )
    done < <(fgrep "real" "${tmp_output_file}")

    ## get physical cores used
    local ncore=$( grep "${ncore_label}" "${tmp_output_file}" | cut -d " " -f 2 )

    ## compute total events processed per core
    local njob=$( grep "${nloop_label}" "${tmp_output_file}" | cut -d " " -f 2 )
    local nproc=$( grep "${nproc_label}" "${tmp_output_file}" | cut -d " " -f 2 )

    ## divide total events by time
    local throughput=$( bc -l <<< "(${njob} * ${nproc}) / ${total_time}" )

    ## dump result into final output file
    echo "${test_label} ${throughput}" >> "${output_file}"
}
export -f DumpIntoFileThroughput
