#!/bin/bash

##########################
## Global Configuration ##
##########################

## Instruction sets defined with "make" command line settings
export SSE3="CPPUSERFLAGS+=\"-march=core2\" CXXUSERFLAGS+=\"-march=core2\" VEC_GCC=\"-march=core2\" VEC_ICC=\"-march=core2\""
export AVX=""
export AVX2="AVX2:=1"
export AVX512="AVX_512:=1"

## Output options
export tmp_ext="log"
export ext="txt"

## Tmp output labels
export nth_label="nTH"
export nev_label="nEV"
export njob_label="nJOB"
export ncore_label="nCORE"
export nproc_label="nPROC"
export nloop_label="nLOOP"

######################
## N Physical Cores ##
######################

function GetNCore ()
{
    local nth=${1}
    local maxcore=${2}

    if (( ${nth} <= ${maxcore} ))
    then
	local ncore="${nth}"
    else
	local ncore="${maxcore}"
    fi
 
    echo "${ncore}"
}
export -f GetNCore

####################
## Core Test Loop ##
####################

function MkFitLoop ()
{
    local min_duration=${1}
    local test_exe=${2}
    local nproc=${3}
    local njob=${4}
    
    local start_time=$( date +"%s" )
    local end_time=$(( ${start_time} + ${min_duration} ))
    
    ## compute number of events to process per job
    local nproc_per_job=$(( ${nproc} / ${njob} ))

    ## global variable to be read back in main loop to keep track of number of times processed
    nloop=0

    ## run stress test for min min_duration with an emulated do-while loop: https://stackoverflow.com/a/16491478
    while

    ## launch jobs in parallel to background : let scheduler put jobs all around
    for (( ijob = 0 ; ijob < ${njob} ; ijob++ ))
    do
	## want each mkFit job to process different events, so compute an offset
	local start_event=$(( ${nproc_per_job} * ${ijob} ))

        ## run the executable
	${test_exe} --num-events ${nproc_per_job} --start-event ${start_event} &
    done

    ## wait for all background processes to finish --> non-ideal as we would rather "stream" jobs launching
    wait
    
    ## increment nloop counter
    ((nloop++))

    ## perform check now to end loop : if current time is greater than projected end time, break.
    local current_time=$( date +"%s" )
    (( ${current_time} <= ${end_time} ))
    do
	continue
    done
}
export -f MkFitLoop

########################################
## Dump Info about Test into Tmp File ##
########################################

function AppendTmpFile ()
{
    local tmp_output_file=${1}
    local ncore=${2}
    local nproc=${3}
    local nloop=${4}

    echo "${ncore_label} ${ncore}" >> "${tmp_output_file}"
    echo "${nproc_label} ${nproc}" >> "${tmp_output_file}"
    echo "${nloop_label} ${nloop}" >> "${tmp_output_file}"
}
export -f AppendTmpFile

####################################
## Dump Tmp Output into Main File ##
####################################

function DumpIntoFile ()
{
    local tmp_output_file=${1}
    local output_file=${2}

    ## get wall-clock time, split 
    read -ra time_arr < <(grep "real" "${tmp_output_file}")
    local tmp_time=${time_arr[1]}

    local mins=$( echo "${tmp_time}" | cut -d "m" -f 1 )
    local secs=$( echo "${tmp_time}" | cut -d "m" -f 2  | cut -d "s" -f 1 )
    
    local total_time=$( bc -l <<< "${mins} * 60 + ${secs}" )
	
    ## get physical cores used
    local ncore=$( grep "${ncore_label}" "${tmp_output_file}" | cut -d " " -f 2 )

    ## compute total events processed per core
    local nloop=$( grep "${nloop_label}" "${tmp_output_file}" | cut -d " " -f 2 )
    local nproc=$( grep "${nproc_label}" "${tmp_output_file}" | cut -d " " -f 2 )

    local total_proc=$(( ${nloop} * ${nproc} )) 
    local total_proc_per_core=$( bc -l <<< "${total_proc} / ${ncore}" )

    ## divide time by total events processed per core 
    local norm_time=$( bc -l <<< "${total_time} / ${total_proc_per_core}" )

    ## dump result into final output file
    echo "${test_label} ${norm_time}" >> "${output_file}"
}
export -f DumpIntoFile
