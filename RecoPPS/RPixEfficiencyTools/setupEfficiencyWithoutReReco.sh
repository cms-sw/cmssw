#!/bin/bash
if [ $# -ne 1 ]
then
	echo "Run Number required. Nothing done..."
else
	export CMSSW_BASE=`readlink -f ../../..`
	eval "mkdir -p -v InputFiles OutputFiles Jobs OutputFiles/PlotsRun${1} LogFiles  test/OutputFiles test/Jobs"
	echo "file:$CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/OutputFiles/Run${1}.root" > "InputFiles/Run${1}.dat"
	eval "ln -s /eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/ReRecoOutputTmp_CMSSW_10_6_2/Run${1}.root test/OutputFiles/Run${1}.root"
	echo "***Creating JSON file***"
	# CHECK THIS!!
	jsonline=`eval "sed -n -e 's/^.*${1}/\"${1}/p' /eos/project/c/ctpps/Operations/DataExternalConditions/2018/CMSgolden_2RPGood_anyarms.json"` 
	echo "{${jsonline}}" | sed -n -e 's/,}/}/p' > "test/JSONFiles/Run${1}.json"
	echo "JSON File contains:"
	eval "more test/JSONFiles/Run${1}.json"
	echo "Saved as test/JSONFiles/Run${1}.json"	
fi