#!/bin/bash

source xeon_scripts/init-env.sh
source xeon_scripts/throughput-test-common.sh

ben_arch=${1} # SNB (phi1), KNL (phi2), SKL-SP (phi3)

if [[ "${ben_arch}" == "KNL" ]]
then
    mOpt="-j 64"
    maxcore=64
    declare -a instruction_sets=(AVX512)
    declare -a thread_combo_arr=("1 1" "2 2" "4 4" "8 8" "16 16" "32 32" "64 64" "128 128" "256 256")
elif [[ "${ben_arch}" == "SKL-SP" ]]
then
    mOpt="-j 32"
    maxcore=32
    declare -a instruction_sets=(AVX512)
    declare -a thread_combo_arr=("1 1" "2 2" "4 4" "8 8" "16 16" "32 32" "64 64")
elif [[ "${ben_arch}" == "LNX-G" ]]
then
    mOpt="-j 32"
    maxcore=32
    declare -a instruction_sets=(AVX512)
    declare -a thread_combo_arr=("1 1" "2 2" "4 4" "8 8" "16 16" "32 32" "64 64")
elif [[ "${ben_arch}" == "LNX-S" ]]
then
    mOpt="-j 32"
    maxcore=32
    declare -a instruction_sets=(AVX512)
    declare -a thread_combo_arr=("1 1" "2 2" "4 4" "8 8" "16 16" "32 32" "64 64")
else
    echo "${ben_arch} is not a valid architecture! Exiting..."
    exit
fi


## Common file setup
dir=/data2/slava77/samples/
subdir=2021/11834.0_TTbar_14TeV+2021/AVE_50_BX01_25ns/
file=memoryFile.fv6.default.211008-c6b7c67.bin
#base_nevents=20 # 7/2 seconds
base_nevents=2000 # 30/10 minutes

## Common mkFit options
seeds="--cmssw-n2seeds"
algo="--build-ce"
opts="--silent --loop-over-file --remove-dup --use-dead-modules --backward-fit"
base_exe="./mkFit/mkFit --input-file ${dir}/${subdir}/${file} ${seeds} ${algo} ${opts}"

## Output options
base_outname="throughput"
output_file="${base_outname}_results.${ext}"

###############
## Run tests ##
###############

## loop instruction sets (i.e. build minimally)
for instruction_set in "${instruction_sets[@]}"
do
    ## compile once, using settings for the given instruction set
    make distclean
    make ${mOpt} ${!instruction_set}

    echo "Ensuring the input file is fully in the memory caches"
    dd if=${dir}/${subdir}/${file} of=/dev/null bs=10M
    dd if=${dir}/${subdir}/${file} of=/dev/null bs=10M
    dd if=${dir}/${subdir}/${file} of=/dev/null bs=10M
    dd if=${dir}/${subdir}/${file} of=/dev/null bs=10M

    ## run thread combo tests (nThreads, nEventsInFlight)
    for thread_combo in "${thread_combo_arr[@]}"
    do echo "${thread_combo}" | while read -r nth nev
	do
	    ## compute total number of events to process
	    ncore=${nth}
	    nproc=$(( ${base_nevents} * ${ncore} ))

	    ## print out which test is being performed
	    test_label="${instruction_set}_${nth_label}${nth}_${nev_label}${nev}"
	    echo "$(date) Running throughput test for: ${test_label}..."

	    ## test executable
	    test_exe="${base_exe} --num-thr ${nth} --num-thr-ev ${nev}"

	    ## output file
	    tmp_output_file="${base_outname}_${test_label}.${tmp_ext}"

	    ## execute test
            MkFitThroughput "${test_exe}" "${nproc}" "1" "${tmp_output_file}"
            cat ${tmp_output_file}.* > ${tmp_output_file}

	    ## add other info about test to tmp file
	    AppendTmpFile "${tmp_output_file}" "${ncore}" "${nproc}" "1"


            ## run a test of N jobs, single thread each
            njob=${nth}

	    ## print out which test is being performed
	    test_label="${instruction_set}_${njob_label}${njob}"
	    echo "$(date) Running throughput test for: ${test_label}..."

	    ## test executable
	    test_exe="${base_exe} --num-thr 1 --num-thr-ev 1"

	    ## output file
	    tmp_output_file="${base_outname}_${test_label}.${tmp_ext}"

	    ## execute test
            MkFitThroughput "${test_exe}" "${base_nevents}" "${njob}" "${tmp_output_file}"
            cat ${tmp_output_file}.* > ${tmp_output_file}

            ## add other info about test to tmp file
	    AppendTmpFile "${tmp_output_file}" "1" "${nproc}" "${njob}"

	done # end loop over reading thread combo
    done # end loop over thread combos

done # end loop over instruction set

#######################
## Make Final Output ##
#######################

## init output file
> "${output_file}"
echo -e "Throughput test meta-data\n" >> "${output_file}"
echo "ben_arch: ${ben_arch}" >> "${output_file}"
echo "base_exe: ${base_exe}" >> "${output_file}"
echo "base_nevents: ${base_nevents}" >> "${output_file}"
echo -e "\nResults (events/s)\n" >> "${output_file}"

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
	    DumpIntoFileThroughput "${tmp_output_file}" "${output_file}"


            njob=${nth}
	    ## get test label, print it
	    test_label="${instruction_set}_${njob_label}${njob}"
	    echo "Computing time for: ${test_label}"

	    ## get tmp output file name
	    tmp_output_file="${base_outname}_${test_label}.${tmp_ext}"

	    ## dump into output file
	    DumpIntoFileThroughput "${tmp_output_file}" "${output_file}"

	done # end loop over reading thread combo
    done # end loop over thread combos
done # end loop over instruction set

#########################################
## Clean up and Restore Default Status ##
#########################################

make distclean

###################
## Final Message ##
###################

echo "$(date) Finished throughput test!"
