if [ $# -ne 1 ]
then
    echo "Run Number required. Nothing done."
else
	export CMSSW_BASE=`readlink -f ../../..`
	echo ""
	echo "***Cleaning Up***"
	echo "Removing /eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/ReRecoOutputTmp_CMSSW_10_6_2/Run${1}.root and its link"
	eval "rm /eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/ReRecoOutputTmp_CMSSW_10_6_2/Run${1}.root ReReco/OutputFiles/Run${1}.root"
	echo "Removing LogFiles"
	eval "rm test/LogFiles/Run${1}.*"
	echo "***Saving in Archive***"
	echo "Saving OutputFiles/Run${1}.root in /eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/Results2018"
	eval "cp OutputFiles/Run${1}.root /eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/Results2018/"
	echo "Saving Plots in /eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/Results2018/PlotsRun${1}"
	eval "cp -r OutputFiles/PlotsRun${1} /eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/Results2018/"
	echo ""
fi