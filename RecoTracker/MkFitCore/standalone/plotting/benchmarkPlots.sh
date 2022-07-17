#! /bin/bash

## input
suite=${1:-"forPR"}
useARCH=${2:-0}
lnxuser=${3:-${USER}}

## In case this is run separately from the main script
source xeon_scripts/common-variables.sh ${suite} ${useARCH} ${lnxuser}
source xeon_scripts/init-env.sh

for archV in "${arch_array_benchmark[@]}" 
do echo ${archV} | while read -r archN archO
    do
	for build in "${ben_builds[@]}"
	do echo ${!build} | while read -r bN bO
	    do
		# see if a test was run for this build routine
		vu_check=$( CheckIfVU ${build} )
		th_check=$( CheckIfTH ${build} )

		echo "Extract benchmarking results for" ${bN} "on" ${archN}
		python plotting/makeBenchmarkPlots.py ${archN} ${sample} ${bN} ${vu_check} ${th_check}
	    done
	done

	echo "Make final plot comparing different build options for" ${archN}
	root -b -q -l plotting/makeBenchmarkPlots.C\(\"${archN}\",\"${sample}\",\"${suite}\"\)
	
	for build in "${meif_builds[@]}"
	do echo ${!build} | while read -r bN bO
	    do
		echo "Extract multiple events in flight benchmark results for" ${bN} "on" ${archN}
		python plotting/makeMEIFBenchmarkPlots.py ${archN} ${sample} ${bN}
		
		echo "Make final plot comparing multiple events in flight for" ${bN} "on" ${archN}
		root -b -q -l plotting/makeMEIFBenchmarkPlots.C\(\"${archN}\",\"${sample}\",\"${bN}\"\)
	    done
	done
    done
done
