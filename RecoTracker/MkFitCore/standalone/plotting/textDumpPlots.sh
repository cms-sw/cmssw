#! /bin/bash

## input
suite=${1:-"forPR"}
useARCH=${2:-0}
lnxuser=${3:-${USER}}

## In case this is run separately from the main script
source xeon_scripts/common-variables.sh ${suite} ${useARCH} ${lnxuser}
source xeon_scripts/init-env.sh

##### Make plots of track properties (kinematics, nHits, etc) from text files, comparing different machine configurations #####
for build in "${text_builds[@]}"
do echo ${!build} | while read -r bN bO
    do
	echo "Making plots from text files for" ${sample} ":" ${bN}
	for archV in "${arch_array_textdump[@]}" 
	do echo ${archV} | while read -r archN archO
	    do
		echo "Extracting plots from dump for" ${archN} ${archO}
		python plotting/makePlotsFromDump.py ${archN} ${sample} ${bN} ${archO}
	    done
	done
		
	echo "Making comparison plots from dump for" ${sample} ":" ${bN}
	root -b -q -l plotting/makePlotsFromDump.C\(\"${sample}\",\"${bN}\",\"${suite}\",${useARCH}\)
    done
done
