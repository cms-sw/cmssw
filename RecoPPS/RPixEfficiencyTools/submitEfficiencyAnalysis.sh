if [ $# -ne 1 ];
then
    echo "Run Number required. Nothing done."
else
	export X509_USER_PROXY=x509up_u93252
	export SCRAM_ARCH=slc7_amd64_gcc700
	export CMSSW_BASE=`readlink -f ../../..`
	if [ -f "Jobs/Run${1}.sub" ]
	then
		echo ""
		echo "The job file already exists"
		echo "To submit the analysis with the already existing file do:"
		echo "condor_submit Jobs/Run${1}.sub"
		echo "Otherwise delete the file and rerun this script to generate a new one"
		echo ""
	else
		echo ""
		eval 'echo "executable			= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/FullEfficiencyAnalysis.sh" 	>> Jobs/Run${1}.sub'
		eval 'echo "arguments  			= ${1} $CMSSW_BASE" 														>> Jobs/Run${1}.sub'
		eval 'echo "output 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/LogFiles/Run${1}.out" 		>> Jobs/Run${1}.sub'
		eval 'echo "error 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/LogFiles/Run${1}.err" 		>> Jobs/Run${1}.sub'
		eval 'echo "log 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/LogFiles/Run${1}.log" 		>> Jobs/Run${1}.sub'
		eval 'echo "+JobFlavour			= \"tomorrow\""																>> Jobs/Run${1}.sub'
		eval 'echo "notify_user			= andrea.bellora@cern.ch"													>> Jobs/Run${1}.sub'
		eval 'echo "notification		= Always"																	>> Jobs/Run${1}.sub'
		eval 'echo "queue"																							>> Jobs/Run${1}.sub'
		
		condor_submit "Jobs/Run${1}.sub" 
		echo "stdout and stderr files are saved in LogFiles/Run${1}.out and LogFiles/Run${1}.err"
		echo "At the end of the job the output will be saved in OutputFiles/Run${1}.root"
		echo ""
		echo "Check the output overview with: root -l 'ViewPlots.C(${1})'"
		echo "If the analysis goes well run: ./cleanAndSave.sh ${1}"
		echo ""
	fi
fi