#!/bin/bash

function find_newest_set {
	local datasets=$1
	local dates=`echo "$1" | grep -o '\-.*\-'`
	local dates=`eval echo '${dates}' | sed 's/-//g'`
	if [[ $dates == *"_"* ]]; 
	then 
		dates=`eval echo $dates | awk -F_ '{print $1}'`
	fi
	local newest_date=`eval echo $dates | awk '{print $1}'`
	local newest_date_conv=`eval date -d "$newest_date" +%Y%m%d`
	for date in ${dates[@]}
	do
		local date_conv=`eval date -d \"$date\" +%Y%m%d`
		if [ $date_conv -ge $newest_date_conv ]
		then
			local newest_date=$date
		fi
	done
	for dataset in ${datasets[@]}
	do
		newest_dataset=`eval echo $dataset | grep ${newest_date}`
		if [[ -z $newest_dataset ]]
		then
			continue
		else
			break
		fi
	done
}

if [ $# -ne 1 ]
then
    echo "Run Number required. Nothing done."
else
	export X509_USER_PROXY=x509up_u93252
	export CMSSW_BASE=`readlink -f ../../..`

	datasets=(
		ZeroBias
		SingleMuon
		EGamma)

	jsonFile=/eos/project/c/ctpps/Operations/DataExternalConditions/2018/CMSgolden_2RPGood_anyarms.json

	
	echo ""
	echo "*****Setting up for Run ${1}*****"
	eval "mkdir -p -v InputFiles OutputFiles OutputFiles/PlotsRun${1} LogFiles Jobs test/JSONFiles test/InputFiles test/OutputFiles test/LogFiles test/Jobs"
	echo ""
	echo "***Creating file list for ReReco***"

	if [ -f "test/InputFiles/Run${1}.dat" ] 
	then
		rm test/InputFiles/Run${1}.dat
		touch test/InputFiles/Run${1}.dat
	else
		touch test/InputFiles/Run${1}.dat

	fi

	for dataset in ${datasets[@]}
	do
		echo "Preparing files for dataset \"${dataset}\""
		set=`eval dasgoclient -query=\"dataset run=${1}\" | grep -i /${dataset}/.*/AOD`
		nsets=`echo "${set}" | wc -l`
		if [[ -z ${set} ]]
		then
			echo "Dataset not found"
			echo ""
		else
			if [ ${nsets} -eq 1 ] 
			then
				echo "Using dataset: ${set}"
			else
				set=`echo "${set}" | grep -v PromptReco`
				nsets=`echo "${set}" | wc -l`
				
				if [ "$set" == "" ]
				then
					echo "Only PromptReco sets found"
					set=`eval dasgoclient -query=\"dataset run=${1}\" | grep -i /${dataset}/.*/AOD | grep 'Run201[0-9][[:upper:]]-PromptReco'`
					nsets=`echo "${set}" | wc -l`
					if [ "${nsets}" -eq 1 ]
					then
						echo "Using dataset: ${set}"
					else
						set=`echo $set | awk '{print $1;}'`
					fi
				else
				if [ "${nsets}" -eq 1 ]
					then
						echo "Using dataset: ${set}"
					else
						echo "More than one dataset was found"
						find_newest_set "${set}"
						set=$newest_dataset
						echo "Using dataset: ${set}"
					fi
				fi
			fi
		
			if [ ${set} == "" ]
			then
				echo "ERROR: NO DATASET FOUND!"
				exit 1
			fi

			fnprev=`eval "more test/InputFiles/Run${1}.dat | wc -l"`
			eval "dasgoclient -query=\"file dataset=${set} run=${1}\" >> test/InputFiles/Run${1}.dat"
			fn=`eval "more test/InputFiles/Run${1}.dat | wc -l"`
			echo "ReReco input files found for dataset \"${dataset}\": $(($fn-$fnprev))"
			echo ""
		fi
	done
	fn=`eval "more test/InputFiles/Run${1}.dat | wc -l"`
	echo "Total ReReco input files found: ${fn}"
	echo "Saved in test/InputFiles/Run${1}.dat"
	echo ""
	echo "***Creating JSON file***"
	# CHECK THIS !!
	jsonline=`eval "sed -n -e 's/^.*${1}/\"${1}/p' ${jsonFile}"`
	echo "{${jsonline}}" | sed -n -e 's/,}/}/p' > "test/JSONFiles/Run${1}.json"
	echo "JSON File contains:"
	eval "more test/JSONFiles/Run${1}.json"
	echo "Saved as test/JSONFiles/Run${1}.json"
	echo ""
	echo "***Setting up Input File for Efficiency Analysis***"
	echo "file:$CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/OutputFiles/Run${1}.root" > "InputFiles/Run${1}.dat"
	echo "Saved as InputFiles/Run${1}.dat"
	echo ""
	echo "***Setting up links to ReReco OutputFiles***"
	eval "ln -s /eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/ReRecoOutputTmp_CMSSW_10_6_2/Run${1}.root test/OutputFiles/Run${1}.root"
	echo ""
	echo "Run: ./submitReReco.sh ${1}"
	echo ""
fi