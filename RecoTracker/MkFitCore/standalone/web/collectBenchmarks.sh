#! /bin/bash

###########
## Input ##
###########

dir=${1:-"benchmarks"}
suite=${2:-"forPR"} # which set of benchmarks to run: full, forPR, forConf
useARCH=${3:-0}
whichcands=${4:-"build"}

###################
## Configuration ##
###################
source xeon_scripts/common-variables.sh ${suite} ${useARCH}
source xeon_scripts/init-env.sh

######################################
## Move Compute Performance Results ##
######################################

# Move subroutine build benchmarks
builddir="Benchmarks"
mkdir -p ${dir}/${builddir}
mkdir -p ${dir}/${builddir}/logx

for ben_arch in "${arch_array[@]}"
do
    for benchmark in TH VU
    do
	oBase=${ben_arch}_${sample}_${benchmark}

	mv ${oBase}_"time".png ${dir}/${builddir}
	mv ${oBase}_"speedup".png ${dir}/${builddir}

	mv ${oBase}_"time_logx".png ${dir}/${builddir}/logx
	mv ${oBase}_"speedup_logx".png ${dir}/${builddir}/logx
    done
done

# Move multiple events in flight plots
meifdir="MultEvInFlight"
mkdir -p ${dir}/${meifdir}
mkdir -p ${dir}/${meifdir}/logx

for ben_arch in "${arch_array[@]}" 
do
    for build in "${meif_builds[@]}"
    do echo ${!build} | while read -r bN bO
	do
            oBase=${ben_arch}_${sample}_${bN}_"MEIF"
	    
            mv ${oBase}_"time".png ${dir}/${meifdir}
            mv ${oBase}_"speedup".png ${dir}/${meifdir}
	    
            mv ${oBase}_"time_logx".png ${dir}/${meifdir}/logx
            mv ${oBase}_"speedup_logx".png ${dir}/${meifdir}/logx
	done
    done
done

# Move plots from text dump
dumpdir="PlotsFromDump"
mkdir -p ${dir}/${dumpdir}
mkdir -p ${dir}/${dumpdir}/diffs

for build in "${text_builds[@]}"
do echo ${!build} | while read -r bN bO
    do
	for var in nHits pt eta phi
	do
	    mv ${sample}_${bN}_${var}.png ${dir}/${dumpdir}
	    mv ${sample}_${bN}_"d"${var}.png ${dir}/${dumpdir}/diffs
	done
    done
done

######################################
## Move Physics Performance Results ##
######################################

# Make SimTrack Validation directories
simdir=("SIMVAL_MTV" "SIMVAL_MTV_SEED")
simval=("SIMVAL" "SIMVALSEED")

for((i=0;i<${#simdir[@]};++i));do

mkdir -p ${dir}/${simdir[i]}
mkdir -p ${dir}/${simdir[i]}/logx
mkdir -p ${dir}/${simdir[i]}/diffs
mkdir -p ${dir}/${simdir[i]}/nHits
mkdir -p ${dir}/${simdir[i]}/score

# Move text file dumps for SimTrack Validation
for build in "${val_builds[@]}"
do echo ${!build} | while read -r bN bO
    do
	vBase=${val_arch}_${sample}_${bN}
	mv "validation"_${vBase}_${simval[i]}/"totals_validation"_${vBase}_${simval[i]}.txt ${dir}/${simdir[i]}
    done
done

# Move dummy CMSSW text file (SimTrack Validation)
vBase=${val_arch}_${sample}_CMSSW
mv validation_${vBase}_${simval[i]}/totals_validation_${vBase}_${simval[i]}.txt ${dir}/${simdir[i]}

# Move rate plots for SimTrack Validation
for rate in eff ineff_brl ineff_trans ineff_ec dr fr
do
    for pt in 0p0 0p9 2p0
    do
	for var in phi eta nLayers
	do 
	    mv ${val_arch}_${sample}_${rate}_${var}_${whichcands}_"pt"${pt}_${simval[i]}.png ${dir}/${simdir[i]}
	done
    done

    # only copy pt > 0 for pt rate plots
    for var in pt pt_zoom
    do 
	mv ${val_arch}_${sample}_${rate}_${var}_${whichcands}_"pt0p0"_${simval[i]}.png ${dir}/${simdir[i]}
    done

    mv ${val_arch}_${sample}_${rate}_"pt_logx"_${whichcands}_"pt0p0"_${simval[i]}.png ${dir}/${simdir[i]}/logx
done

# Move kinematic diff plots for SimTrack Validation
for coll in bestmatch allmatch
do 
    for var in nHits invpt phi eta
    do
	for pt in 0p0 0p9 2p0
	do
	    mv ${val_arch}_${sample}_${coll}_"d"${var}_${whichcands}_"pt"${pt}_${simval[i]}.png ${dir}/${simdir[i]}/diffs
	done
    done
done

# Move track quality plots for SimTrack Validation (nHits,score)
for coll in allreco fake bestmatch allmatch
do 
    for pt in 0p0 0p9 2p0
    do
	for qual in nHits score
	do
	    mv ${val_arch}_${sample}_${coll}_${qual}_${whichcands}_"pt"${pt}_${simval[i]}.png ${dir}/${simdir[i]}/${qual}
	done
    done
done
done

# Final message
echo "Finished collecting benchmark plots into ${dir}!"
