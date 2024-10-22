#!/bin/bash

###############################################################################
##                                    README!                                ##
##                                                                           ##
## Stress test script to run on phiN, testing different thread/MEIF combos   ##
## with different instruction set architecture extensions, using default     ## 
## settings of benchmarking scripts for clone engine track finding + CMSSW   ##
## n2-seeding, input sample ttbar PU70.                                      ##
##                                                                           ##
## Can vary thread/MEIF combos, input file, seeds, building algo by editting ##
## this script manually.                                                     ##
##                                                                           ##
## Command line inputs are which platform to stress (ben_arch), enable       ##
## TurboBoost OFF/ON (no_turbo), the min time per test (min_duration), the   ##   
## time between each test (sleep_time), and the number of events to process  ## 
## per physical core (base_events).                                          ##
##                                                                           ##
## N.B.: base_events MUST be a number divisible by 4! This is because the    ##
## max physical cores on KNL is 64, but the highest nTH/nJOB test is 256.    ##       
##                                                                           ##
## Output file lists stress test time per event processed per physical core. ##
###############################################################################

########################
## Source Environment ##
########################

source xeon_scripts/init-env.sh
source xeon_scripts/stress-test-common.sh

###################
## Configuration ##
###################

## Command line inputs
ben_arch=${1} # SNB (phi1), KNL (phi2), SKL-SP (phi3)
no_turbo=${2:-1} # Turbo OFF or ON --> default is OFF!
min_duration=${3:-1800} # min time spent for each test [s]
sleep_time=${4:-300} # sleep time between tests [s]
base_nevents=${5:-120} # number of events to process per physical core, must be divisible by 4

## platform specific settings
if [[ "${ben_arch}" == "SNB" ]]
then
    mOpt="-j 12"
    maxcore=12
    declare -a instruction_sets=(SSE3 AVX)
    declare -a thread_combo_arr=("1 1" "6 6" "12 6" "12 12" "24 6" "24 12" "24 24")
    declare -a njob_arr=("12" "24")
elif [[ "${ben_arch}" == "KNL" ]]
then
    mOpt="-j 64"
    maxcore=64
    declare -a instruction_sets=(SSE3 AVX AVX2 AVX512)
    declare -a thread_combo_arr=("1 1" "32 32" "64 32" "64 64" "128 32" "128 64" "128 128" "256 32" "256 64" "256 128" "256 256")
    declare -a njob_arr=("32" "64" "128" "256")
elif [[ "${ben_arch}" == "SKL-SP" ]]
then
    mOpt="-j 32"
    maxcore=32
    declare -a instruction_sets=(SSE3 AVX AVX2 AVX512)
    declare -a thread_combo_arr=("1 1" "16 16" "32 16" "32 32" "48 16" "48 32" "64 16" "64 32" "64 64")
    declare -a njob_arr=("32" "64")
else 
    echo "${ben_arch} is not a valid architecture! Exiting..."
    exit
fi

## Common file setup
dir=/data2/slava77/samples/
subdir=2021/11834.0_TTbar_14TeV+2021/AVE_50_BX01_25ns/
file=memoryFile.fv6.default.211008-c6b7c67.bin

## Common mkFit options
seeds="--cmssw-n2seeds"
algo="--build-ce"
opts="--silent --remove-dup --use-dead-modules --backward-fit"
base_exe="./mkFit/mkFit --input-file ${dir}/${subdir}/${file} ${seeds} ${algo} ${opts}"

## Output options
base_outname="stress_test"
output_file="${base_outname}_results.${ext}"

## Set TurboBoost option
echo "${no_turbo}" | PATH=/bin sudo /usr/bin/tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null 2>&1  

###############
## Run tests ##
###############

## loop instruction sets (i.e. build minimally)
for instruction_set in "${instruction_sets[@]}"
do
    ## compile once, using settings for the given instruction set
    make distclean
    make ${mOpt} ${!instruction_set}
    
    ## run thread combo tests (nThreads, nEventsInFlight)
    for thread_combo in "${thread_combo_arr[@]}"
    do echo "${thread_combo}" | while read -r nth nev
	do
	    ## compute total number of events to process
	    ncore=$( GetNCore "${nth}" "${maxcore}" ) 
	    nproc=$(( ${base_nevents} * ${ncore} ))

	    ## print out which test is being performed
	    test_label="${instruction_set}_${nth_label}${nth}_${nev_label}${nev}"
	    echo "Running stress test for: ${test_label}..."

	    ## test executable
	    test_exe="${base_exe} --num-thr ${nth} --num-thr-ev ${nev}"

	    ## output file
	    tmp_output_file="${base_outname}_${test_label}.${tmp_ext}"
	    
	    ## execute test and pipe time to output file: https://stackoverflow.com/a/2409214
	    { time MkFitLoop "${min_duration}" "${test_exe}" "${nproc}" "1" > /dev/null 2>&1 ; } 2> "${tmp_output_file}"

	    ## pause to let machine cool down between each test
	    sleep "${sleep_time}"

	    ## add other info about test to tmp file
	    AppendTmpFile "${tmp_output_file}" "${ncore}" "${nproc}" "${nloop}"
	done # end loop over reading thread combo
    done # end loop over thread combos

    ## run special test of N jobs, single thread each
    for njob in "${njob_arr[@]}"
    do
	## compute total number of events to process
	ncore=$( GetNCore "${njob}" "${maxcore}" ) 
	nproc=$(( ${base_nevents} * ${ncore} ))

	## print out which test is being performed
	test_label="${instruction_set}_${njob_label}${njob}"
	echo "Running stress test for: ${test_label}..."

	## test executable
	test_exe="${base_exe} --num-thr 1 --num-thr-ev 1"

	## output file
	tmp_output_file="${base_outname}_${test_label}.${tmp_ext}"
	    
	## execute test and pipe time to output file: https://stackoverflow.com/a/2409214
	{ time MkFitLoop "${min_duration}" "${test_exe}" "${nproc}" "${njob}" > /dev/null 2>&1 ; } 2> "${tmp_output_file}"

        ## add other info about test to tmp file
	AppendTmpFile "${tmp_output_file}" "${ncore}" "${nproc}" "${nloop}"

	## pause to let machine cool down between each test
	sleep "${sleep_time}"
    done # end loop over njob for single thread

done # end loop over instruction set

#######################
## Make Final Output ##
#######################

## init output file
> "${output_file}"
echo -e "Stress test meta-data\n" >> "${output_file}"
echo "ben_arch: ${ben_arch}" >> "${output_file}"
echo "no_turbo: ${no_turbo}" >> "${output_file}"
echo "min_duration [s]: ${min_duration}" >> "${output_file}"
echo "sleep_time [s]: ${sleep_time}" >> "${output_file}"
echo "base_exe: ${base_exe}" >> "${output_file}"
echo "base_nevents: ${base_nevents}" >> "${output_file}"
echo -e "\nResults\n" >> "${output_file}"

## loop over all output files, and append results to single file
for instruction_set in "${instruction_sets[@]}"
do
    ## loop over nThread/MEIF tests, and append to single file
    for thread_combo in "${thread_combo_arr[@]}"
    do echo "${thread_combo}" | while read -r nth nev
	do
	    ## get test label, print it
	    test_label="${instruction_set}_${nth_label}${nth}_${nev_label}${nev}"
	    echo "Computing time for: ${test_label}"
	    
            ## get tmp output file name
	    tmp_output_file="${base_outname}_${test_label}.${tmp_ext}"
	    
	    ## dump into output file
	    DumpIntoFile "${tmp_output_file}" "${output_file}"
	done # end loop over reading thread combo
    done # end loop over thread combos

    ## loop over single thread njob tests, and append to single file
    for njob in "${njob_arr[@]}"
    do
	## get test label, print it
	test_label="${instruction_set}_${njob_label}${njob}"
	echo "Computing time for: ${test_label}"
	
	## get tmp output file name
	tmp_output_file="${base_outname}_${test_label}.${tmp_ext}"
	
	## dump into output file
	DumpIntoFile "${tmp_output_file}" "${output_file}"
    done # end loop over njob array

done # end loop over instruction set

#########################################
## Clean up and Restore Default Status ##
#########################################

make distclean
echo 1 | PATH=/bin sudo /usr/bin/tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null 2>&1

###################
## Final Message ##
###################

echo "Finished stress test!"
