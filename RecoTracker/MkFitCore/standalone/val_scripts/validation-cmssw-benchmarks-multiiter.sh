#! /bin/bash

###########
## Input ##
###########

suite=${1:-"forConf"} # which set of benchmarks to run: full, forPR, forConf, val, valMT1
style=${2:-"--mtv-like-val"} # option --mtv-like-val
inputBin=${3:-"112X_TTbar_PU50_MULTI"}

###################
## Configuration ##
###################

source xeon_scripts/common-variables.sh ${suite}
source xeon_scripts/init-env.sh
export MIMI="CE mimi"
declare -a val_builds=(MIMI)
nevents=250
extraargs="--use-dead-modules"
numiters=10

## Common file setup
case ${inputBin} in 
"104XPU50CCC_MULTI")
        echo "Inputs from 2018 initialStep/default PU 50 with CCC with multiple iterations and hit binary mask"
        dir=/data2/slava77/analysis/CMSSW_10_4_0_patch1_mkFit/pass-df52fcc
        subdir=/initialStep/default/11024.0_TTbar_13/AVE_50_BX01_25ns/RAW4NT  
        file=/memoryFile.fv5.clean.writeAll.CCC1620.recT.allSeeds.masks.201023-64302e5.bin
        ;;
"112X_TTbar_PU50_MULTI")
        echo "Inputs from 2021 TTbar (PU50) sample with multiple iterations and hit binary mask"
        dir=/data2/slava77/samples/
        subdir=2021/11834.0_TTbar_14TeV+2021/AVE_50_BX01_25ns/
        file=memoryFile.fv6.default.211008-c6b7c67.bin
        ;;
"112X_10mu_MULTI")
        echo "Inputs from 2021 10mu sample with multiple iterations and hit binary mask"
        dir=/data2/slava77/samples
        subdir=2021/10muPt0p2to1000HS
        file=memoryFile.fv6.default.211008-c6b7c67.bin
        nevents=20000
        sample=10mu
        ;;
"TTbar_phase2")
        dir=/home/matevz/mic-dev
        subdir=
        file=ttbar-p2.bin
        nevents=100
        extraargs="--geom CMS-phase2"
        numiters=1
        # Pass MIMI flag to ROOT plotting functions -- some will insert STD
        # that does not work with MIMI at this point.
        export MKFIT_MIMI=1
        ;;
*)
        echo "INPUT BIN IS UNKNOWN"
        exit 12
        ;;
esac

## Common executable setup
if [[ `lsb_release -si` == "Fedora" ]]
then
    maxth=16 # 64
    maxvu=8  # 16
    maxev=16 # 32
    avxmakearg="AVX2:=1"
else
    maxth=64
    maxvu=16
    maxev=32
    avxmakearg="AVX_512:=1"
fi

if [[  "${suite}" == "valMT1" ]]
then
    maxth=1
    maxev=1
fi
seeds="--cmssw-n2seeds"
exe="./mkFit --silent ${seeds} --num-thr ${maxth} --num-thr-ev ${maxev} --input-file ${dir}/${subdir}/${file} --num-events ${nevents} --remove-dup ${extraargs}"

## Common output setup
tmpdir="tmp"
base=${val_arch}_${sample}

## flag to save sim info for matched tracks since track states not read in
siminfo="--try-to-save-sim-info"

## backward fit flag
bkfit="--backward-fit"

## validation options: SIMVAL == sim tracks as reference, CMSSWVAL == cmssw tracks as reference
SIMVAL="SIMVAL --sim-val ${siminfo} ${bkfit} ${style} --num-iters-cmssw ${numiters}"
SIMVAL_SEED="SIMVALSEED --sim-val ${siminfo} ${bkfit} --mtv-require-seeds --num-iters-cmssw ${numiters}"

declare -a vals=(SIMVAL SIMVAL_SEED)

## plotting options
SIMPLOT="SIMVAL all 0 0 1"
SIMPLOTSEED="SIMVALSEED all 0 0 1"
SIMPLOT4="SIMVAL iter4 0 4 0"
SIMPLOTSEED4="SIMVALSEED iter4 0 4 0" 
SIMPLOT22="SIMVAL iter22 0 22 0"
SIMPLOTSEED22="SIMVALSEED iter22 0 22 0"
SIMPLOT23="SIMVAL iter23 0 23 0"
SIMPLOTSEED23="SIMVALSEED iter23 0 23 0"
SIMPLOT5="SIMVAL iter5 0 5 0"
SIMPLOTSEED5="SIMVALSEED iter5 0 5 0"
SIMPLOT24="SIMVAL iter24 0 24 0"
SIMPLOTSEED24="SIMVALSEED iter24 0 24 0"
SIMPLOT7="SIMVAL iter7 0 7 0"
SIMPLOTSEED7="SIMVALSEED iter7 0 7 0"
SIMPLOT8="SIMVAL iter8 0 8 0"
SIMPLOTSEED8="SIMVALSEED iter8 0 8 0"
SIMPLOT9="SIMVAL iter9 0 9 0"
SIMPLOTSEED9="SIMVALSEED iter9 0 9 0"
SIMPLOT10="SIMVAL iter10 0 10 0"
SIMPLOTSEED10="SIMVALSEED iter10 0 10 0"
SIMPLOT6="SIMVAL iter6 0 6 0"
SIMPLOTSEED6="SIMVALSEED iter6 0 6 0"

if [[ "${inputBin}" == "TTbar_phase2" ]]
then
    declare -a plots=(SIMPLOT4 SIMPLOTSEED4)
else
    declare -a plots=(SIMPLOT4 SIMPLOTSEED4 SIMPLOT22 SIMPLOTSEED22 SIMPLOT23 SIMPLOTSEED23 SIMPLOT5 SIMPLOTSEED5 SIMPLOT24 SIMPLOTSEED24 SIMPLOT7 SIMPLOTSEED7 SIMPLOT8 SIMPLOTSEED8 SIMPLOT9 SIMPLOTSEED9 SIMPLOT10 SIMPLOTSEED10 SIMPLOT6 SIMPLOTSEED6)
fi

## special cmssw dummy build
CMSSW="CMSSW cmssw SIMVAL --sim-val-for-cmssw ${siminfo} --read-cmssw-tracks ${style} --num-iters-cmssw ${numiters}"
CMSSW2="CMSSW cmssw SIMVALSEED --sim-val-for-cmssw ${siminfo} --read-cmssw-tracks --mtv-require-seeds --num-iters-cmssw ${numiters}"

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
    local iter=${5}
    local cancel=${6}     
    local rmsuff=${7}

    echo "Computing observables for: ${base} ${bN} ${pN} ${p0} ${iter} ${cancel}"
    bExe="root -b -q -l plotting/runValidation.C(\"_${base}_${bN}_${pN}\",${pO},${iter},${cancel},${rmsuff})"
    echo ${bExe}

    ${bExe} || (echo "Crashed on CMD: "${bExe}; exit 3)
}

########################
## Run the validation ##
########################

## Compile once
make distclean
mVal="-j 32 WITH_ROOT:=1 ${avxmakearg}"
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
mv tmp/valtree_*.root .
rm -rf ${tmpdir}


## Compute observables and make images
for plot in "${plots[@]}"
do echo ${!plot} | while read -r pN suff pO iter cancel
    do
	rmsuff=0 # use iterX suffix for output directory
        ## Compute observables for special dummy CMSSW
	if [[ "${pN}" == "SIMVAL" || "${pN}" == "SIMVAL_"* ]]
	then
	    echo ${CMSSW} | while read -r bN bO val_extras
	    do
		plotVal "${base}" "${bN}" "${pN}" "${pO}" "${iter}" "${cancel}" "${rmsuff}"
	    done
	fi
	if [[ "${pN}" == "SIMVALSEED"* ]]
	then
	    echo ${CMSSW2} | while read -r bN bO val_extras
	    do
		plotVal "${base}" "${bN}" "${pN}" "${pO}" "${iter}" "${cancel}" "${rmsuff}"
	    done
	fi

	## Compute observables for builds chosen 
	for build in "${val_builds[@]}"
	do echo ${!build} | while read -r bN bO
	    do
		plotVal "${base}" "${bN}" "${pN}" "${pO}" "${iter}" "${cancel}" "${rmsuff}"
	    done
	done
	
	## overlay histograms
	echo "Overlaying histograms for: ${base} ${vN}"
        if [[  "${suff}" == "all" ]]
        then
            root -b -q -l plotting/makeValidation.C\(\"${base}\",\"_${pN}\",${pO},\"${suite}\"\)
        else
            root -b -q -l plotting/makeValidation.C\(\"${base}\",\"_${pN}_${suff}\",${pO},\"${suite}\"\)
        fi
    done
done

## Final message
echo "Finished physics validation!"
