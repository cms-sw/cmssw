#! /bin/bash

###########
## Input ##
###########

ben_arch=${1} # SNB, KNL, SKL-SP
suite=${2:-"forPR"} # which set of benchmarks to run: full, forPR, forConf
useARCH=${3:-0}
lnxuser=${4:-${USER}}

###################
## Configuration ##
###################

## Source environment and common variables
source xeon_scripts/common-variables.sh ${suite} ${useARCH} ${lnxuser}
source xeon_scripts/init-env.sh

## Platform specific settings
if [[ "${ben_arch}" == "SNB" ]]
then
    mOpt="-j 12"
    maxth=24
    maxvu=8
    declare -a nths=("1" "2" "4" "6" "8" "12" "16" "20" "24")
    declare -a nvus=("1" "2" "4" "8")
    declare -a nevs=("1" "2" "4" "8" "12")
elif [[ "${ben_arch}" == "KNL" ]]
then
    mOpt="-j 64 AVX_512:=1"
    maxth=256
    maxvu=16
    declare -a nths=("1" "2" "4" "8" "16" "32" "64" "96" "128" "160" "192" "224" "256")
    declare -a nvus=("1" "2" "4" "8" "16")
    declare -a nevs=("1" "2" "4" "8" "16" "32" "64" "128")
elif [[ "${ben_arch}" == "SKL-SP" ]]
then
    mOpt="-j 32 AVX_512:=1"
    maxth=64
    maxvu=16
    declare -a nths=("1" "2" "4" "8" "16" "32" "48" "64")
    declare -a nvus=("1" "2" "4" "8" "16")
    declare -a nevs=("1" "2" "4" "8" "16" "32" "64")
elif [[ "${ben_arch}" == "LNX-G" ]]
then 
    mOpt="-j 32 AVX_512:=1"
    maxth=64
    maxvu=16
    declare -a nths=("1" "2" "4" "8" "16" "32" "48" "64")
    declare -a nvus=("1" "2" "4" "8" "16")
    declare -a nevs=("1" "2" "4" "8" "16" "32" "64")
elif [[ "${ben_arch}" == "LNX-S" ]]
then 
    mOpt="-j 32 AVX_512:=1"
    maxth=64
    maxvu=16
    declare -a nths=("1" "2" "4" "8" "16" "32" "48" "64")
    declare -a nvus=("1" "2" "4" "8" "16")
    declare -a nevs=("1" "2" "4" "8" "16" "32" "64")
else 
    echo ${ben_arch} "is not a valid architecture! Exiting..."
    exit
fi

## Common file setup
dir=/data2/slava77/samples/
subdir=2021/11834.0_TTbar_14TeV+2021/AVE_50_BX01_25ns/
file=memoryFile.fv6.default.211008-c6b7c67.bin
nevents=20

## Common executable setup
minth=1
minvu=1
seeds="--cmssw-n2seeds"
exe="./mkFit/mkFit ${seeds} --input-file ${dir}/${subdir}/${file}"

## Common output setup
dump=DumpForPlots
base=${ben_arch}_${sample}

####################
## Run Benchmarks ##
####################

## compile with appropriate options
make distclean ${mOpt}
make ${mOpt}

## Parallelization Benchmarks
for nth in "${nths[@]}"
do
    for build in "${th_builds[@]}"
    do echo ${!build} | while read -r bN bO
	do
	    ## Base executable
	    oBase=${base}_${bN}
	    bExe="${exe} --build-${bO} --num-thr ${nth}"

	    ## Building-only benchmark
	    echo "${oBase}: Benchmark [nTH:${nth}, nVU:${maxvu}int]"
	    ${bExe} --num-events ${nevents} >& log_${oBase}_NVU${maxvu}int_NTH${nth}.txt

	    ## Multiple Events in Flight benchmark
	    check_meif=$( CheckIfMEIF ${build} )
	    if [[ "${check_meif}" == "true" ]]
	    then
		for nev in "${nevs[@]}"
		do
		    if (( ${nev} <= ${nth} ))
		    then
			nproc=$(( ${nevents} * ${nev} ))
			echo "${oBase}: Benchmark [nTH:${nth}, nVU:${maxvu}int, nEV:${nev}]"
			${bExe} --silent --num-thr-ev ${nev} --num-events ${nproc} --remove-dup --use-dead-modules --backward-fit >& log_${oBase}_NVU${maxvu}int_NTH${nth}_NEV${nev}.txt
		    fi
		done
	    fi

	    ## nHits validation
	    check_text=$( CheckIfText ${build} )
	    if (( ${nth} == ${maxth} )) && [[ "${check_text}" == "true" ]]
	    then
		echo "${oBase}: Text dump for plots [nTH:${nth}, nVU:${maxvu}int]"
		${bExe} --dump-for-plots --quality-val --read-cmssw-tracks --num-events ${nevents} --remove-dup --use-dead-modules --backward-fit >& log_${oBase}_NVU${maxvu}int_NTH${nth}_${dump}.txt
	    fi
	done
    done
done

## Vectorization Benchmarks
for nvu in "${nvus[@]}"
do
    make clean ${mOpt}
    make ${mOpt} USE_INTRINSICS:=-DMPT_SIZE=${nvu}

    for build in "${vu_builds[@]}"
    do echo ${!build} | while read -r bN bO
	do
	    ## Common base executable
	    oBase=${base}_${bN}
	    bExe="${exe} --build-${bO} --num-thr ${minth} --num-events ${nevents}"

	    ## Building-only benchmark
	    echo "${oBase}: Benchmark [nTH:${minth}, nVU:${nvu}]"
	    ${bExe} >& log_${oBase}_NVU${nvu}_NTH${minth}.txt

	    ## nHits validation
	    check_text=$( CheckIfText ${build} )
	    if (( ${nvu} == ${minvu} )) && [[ "${check_text}" == "true" ]]
	    then
		echo "${oBase}: Text dump for plots [nTH:${minth}, nVU:${nvu}]"
		${bExe} --dump-for-plots --quality-val --read-cmssw-tracks >& log_${oBase}_NVU${nvu}_NTH${minth}_${dump}.txt
	    fi
	done
    done
done

## Final cleanup
make distclean ${mOpt}

## Final message
echo "Finished compute benchmarks on ${ben_arch}!"
