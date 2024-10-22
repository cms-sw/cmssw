#! /bin/bash

###########
## Input ##
###########

suite=${1:-"forPR"} # which set of benchmarks to run: full, forPR, forConf, val, valMT1
style=${2:-"--mtv-like-val"} # option --mtv-like-val
inputBin=${3:-"112X_TTbar_PU50_MULTI"}

###################
## Configuration ##
###################

source xeon_scripts/common-variables.sh ${suite}
source xeon_scripts/init-env.sh

nevents=500

## Common file setup
case ${inputBin} in 
"91XPU70CCC")
        echo "Inputs from 2017 initialStep PU 70 with CCC -- DO NOT WORK ANYMORE"
        exit 1
        dir=/data2/slava77/samples/2017/pass-c93773a/initialStep
        subdir=PU70HS/10224.0_TTbar_13+TTbar_13TeV_TuneCUETP8M1_2017PU_GenSimFullINPUT+DigiFullPU_2017PU+RecoFullPU_2017PU+HARVESTFullPU_2017PU
        file=memoryFile.fv3.clean.writeAll.CCC1620.recT.082418-25daeda.bin
        ;;
"104XPU50CCC")
        echo "Inputs from 2018 initialStep/default PU 50 with CCC"
        dir=/data2
        subdir=
        file=pu50-ccc-hs.bin
        ;;
"112X_TTbar_PU50_MULTI")
        echo "Inputs from 2021 TTbar (PU50) sample with multiple iterations and hit binary mask"
        dir=/data2/slava77/samples/
        subdir=2021/11834.0_TTbar_14TeV+2021/AVE_50_BX01_25ns/
        file=memoryFile.fv6.default.211008-c6b7c67.bin
        ;;
"104X10muCCC")
        echo "Inputs from 2018 10mu large pt range using the offline initialStep seeds with CCC (phi3)"
        dir=/data2/slava77/samples/2018/pass-925bb57
        subdir=initialStep/default/10muPt0p2to1000HS
        file=memoryFile.fv4.clean.writeAll.CCC1620.recT.191108-c41a0f2.bin
        nevents=10000
        sample=CMSSW_10mu
        ;;
"112X_10mu_MULTI")
        echo "Inputs from 2021 10mu sample with multiple iterations and hit binary mask"
        dir=/data2/slava77/samples
        subdir=2021/10muPt0p2to1000HS
        file=memoryFile.fv6.default.211008-c6b7c67.bin
        nevents=20000
        sample=10mu
        ;;
"104X10muHLT3CCC")
        echo "Inputs from 2018 10mu large pt range using HLT iter0 seeds as triplets with CCC (phi3)"
        dir=/data2/slava77/samples/2018/pass-2eaa1f7
        subdir=hltIter0/default/triplet/10muPt0p2to1000HS
        file=memoryFile.fv4.clean.writeAll.CCC1620.recT.200122-fcff8a8.bin
        nevents=10000
        sample=CMSSW_10mu_HLT3
        ;;
"104X10muHLT4CCC")
        echo "Inputs from 2018 10mu large pt range using HLT iter0 seeds as quadruplets with CCC (phi3)"
        dir=/data2/slava77/samples/2018/pass-2eaa1f7
        subdir=hltIter0/default/quadruplet/10muPt0p2to1000HS
        file=memoryFile.fv4.clean.writeAll.CCC1620.recT.200122-fcff8a8.bin
        nevents=10000
        sample=CMSSW_10mu_HLT4
        ;;
"104XPU50HLT3CCC")
        echo "Inputs from 2018 ttbar PU50 using HLT iter0 seeds as triplets with CCC (phi3)"
        dir=/data2/slava77/samples/2018/pass-2eaa1f7
        subdir=hltIter0/default/triplet/11024.0_TTbar_13/AVE_50_BX01_25ns
        file=memoryFile.fv4.clean.writeAll.CCC1620.recT.200122-fcff8a8.bin
        sample=CMSSW_TTbar_PU50_HLT3
        ;;
"104XPU50HLT4CCC")
        echo "Inputs from 2018 ttbar PU50 using HLT iter0 seeds as quadruplets with CCC (phi3)"
        dir=/data2/slava77/samples/2018/pass-2eaa1f7
        subdir=hltIter0/default/quadruplet/11024.0_TTbar_13/AVE_50_BX01_25ns
        file=memoryFile.fv4.clean.writeAll.CCC1620.recT.200122-fcff8a8.bin
        sample=CMSSW_TTbar_PU50_HLT4
        ;;
*)
        echo "INPUT BIN IS UNKNOWN"
        exit 12
        ;;
esac

## Common executable setup
maxth=64
maxvu=16
maxev=32
if [[  "${suite}" == "valMT1" ]]
then
    maxth=1
    maxev=1
fi
seeds="--cmssw-n2seeds"
exe="./mkFit/mkFit --silent ${seeds} --num-thr ${maxth} --num-thr-ev ${maxev} --input-file ${dir}/${subdir}/${file} --num-events ${nevents} --remove-dup --use-dead-modules"

## Common output setup
tmpdir="tmp"
base=${val_arch}_${sample}

## flag to save sim info for matched tracks since track states not read in
siminfo="--try-to-save-sim-info"

## backward fit flag
bkfit="--backward-fit"

## validation options: SIMVAL == sim tracks as reference, CMSSWVAL == cmssw tracks as reference
SIMVAL="SIMVAL --sim-val ${siminfo} ${bkfit} ${style}"
SIMVAL_SEED="SIMVALSEED --sim-val ${siminfo} ${bkfit} --mtv-require-seeds"
declare -a vals=(SIMVAL SIMVAL_SEED)

## plotting options
SIMPLOT="SIMVAL 0"
SIMPLOTSEED="SIMVALSEED 0"
declare -a plots=(SIMPLOT SIMPLOTSEED)

## special cmssw dummy build
CMSSW="CMSSW cmssw SIMVAL --sim-val-for-cmssw ${siminfo} --read-cmssw-tracks ${style} --num-iters-cmssw 1"
CMSSW2="CMSSW cmssw SIMVALSEED --sim-val-for-cmssw ${siminfo} --read-cmssw-tracks --mtv-require-seeds --num-iters-cmssw 1"

###############
## Functions ##
###############

## validation function
function doVal()
{
    local bN=${1}
    local bO=${2}
    local vN=${3}
    local vO=${4}

    local oBase=${val_arch}_${sample}_${bN}
    local bExe="${exe} ${vO} --build-${bO}"
    
    echo "${oBase}: ${vN} [nTH:${maxth}, nVU:${maxvu}int, nEV:${maxev}]"
    ${bExe} >& log_${oBase}_NVU${maxvu}int_NTH${maxth}_NEV${maxev}_${vN}.txt || (echo "Crashed on CMD: "${bExe}; exit 2)
    
    if (( ${maxev} > 1 ))
    then
        # hadd output files from different threads for this test, then move to temporary directory
        hadd -O valtree.root valtree_*.root
        rm valtree_*.root
    fi
    mv valtree.root ${tmpdir}/valtree_${oBase}_${vN}.root
}		

## plotting function
function plotVal()
{
    local base=${1}
    local bN=${2}
    local pN=${3}
    local pO=${4}
    local iter=${5} # only initialStep
    local cancel=${6}

    echo "Computing observables for: ${base} ${bN} ${pN}"
    bExe="root -b -q -l plotting/runValidation.C(\"_${base}_${bN}_${pN}\",${pO},${iter},${cancel})"
    ${bExe} || (echo "Crashed on CMD: "${bExe}; exit 3)
}

########################
## Run the validation ##
########################

## Compile once
make clean
mVal="-j 32 WITH_ROOT:=1 AVX_512:=1"
make ${mVal}
mkdir -p ${tmpdir}

## Special simtrack validation vs cmssw tracks
echo ${CMSSW} | while read -r bN bO vN vO
do
    doVal "${bN}" "${bO}" "${vN}" "${vO}"
done
## Special simtrack validation vs cmssw tracks
echo ${CMSSW2} | while read -r bN bO vN vO
do
    doVal "${bN}" "${bO}" "${vN}" "${vO}"
done

## Run validation for standard build options
for val in "${vals[@]}"
do echo ${!val} | while read -r vN vO
    do
	for build in "${val_builds[@]}"
	do echo ${!build} | while read -r bN bO
	    do
		doVal "${bN}" "${bO}" "${vN}" "${vO}"
	    done
	done
    done
done

## clean up
make clean ${mVal}
mv tmp/valtree_*.root .
rm -rf ${tmpdir}

## Compute observables and make images
for plot in "${plots[@]}"
do echo ${!plot} | while read -r pN pO
    do
        ## Compute observables for special dummy CMSSW
	if [[ "${pN}" == "SIMVAL" ]]
	then
	    echo ${CMSSW} | while read -r bN bO val_extras
	    do
		iter=4 # only initialStep
		cancel=1
		plotVal "${base}" "${bN}" "${pN}" "${pO}" "${iter}" "${cancel}"
	    done
	fi
	if [[ "${pN}" == "SIMVALSEED" ]]
	then
	    echo ${CMSSW2} | while read -r bN bO val_extras
	    do
		iter=4 # only initialStep
		cancel=1
		plotVal "${base}" "${bN}" "${pN}" "${pO}" "${iter}" "${cancel}"
	    done
	fi

	## Compute observables for builds chosen 
	for build in "${val_builds[@]}"
	do echo ${!build} | while read -r bN bO
	    do
		iter=0
		cancel=1
		plotVal "${base}" "${bN}" "${pN}" "${pO}" "${iter}" "${cancel}"
		#plotVal "${base}" "${bN}" "${pN}" "${pO}"
	    done
	done
	
	## overlay histograms
	echo "Overlaying histograms for: ${base} ${vN}"
	root -b -q -l plotting/makeValidation.C\(\"${base}\",\"_${pN}\",${pO},\"${suite}\"\)
    done
done

## Final cleanup
make distclean ${mVal}

## Final message
echo "Finished physics validation!"
