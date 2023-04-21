if [ $# -ne 1 ]
then
    echo "Run Number required. Nothing done."
else
	# Maximum number of events to be analyzed
	maxEvents=5000000


	eventsPerJob=100000
	export X509_USER_PROXY=x509up_u93252
	export SCRAM_ARCH=slc7_amd64_gcc700
	export CMSSW_BASE=`readlink -f ../../..`

	if [ -f "test/Jobs/Run${1}.dag" ]
	then
		eval 'ls test/Jobs/ | grep "Run${1}.*" | sed -e "s/^/test\/Jobs\//" | xargs rm' # removes all Job files related to DAG
		# rm "test/Jobs/Run${1}.dag"
	fi

	outputFile=/eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/ReRecoOutputTmp_CMSSW_10_6_2/Run${1}.root

	eval 'echo "executable			= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/mergeFiles.sh"						>> test/Jobs/Run${1}_mergeJob.sub'
	eval 'echo "arguments  			= ${1} ${outputFile} $CMSSW_BASE"													>> test/Jobs/Run${1}_mergeJob.sub'
	eval 'echo "output 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/LogFiles/Run${1}_mergeJob.out"	>> test/Jobs/Run${1}_mergeJob.sub'
	eval 'echo "error 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/LogFiles/Run${1}_mergeJob.err"	>> test/Jobs/Run${1}_mergeJob.sub'
	eval 'echo "log 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/LogFiles/Run${1}_mergeJob.log"	>> test/Jobs/Run${1}_mergeJob.sub'
	eval 'echo "notify_user			= andrea.bellora@cern.ch"															>> test/Jobs/Run${1}_mergeJob.sub'
	eval 'echo "notification		= Always"																			>> test/Jobs/Run${1}_mergeJob.sub'
	eval 'echo "use_x509userproxy   = true"																				>> test/Jobs/Run${1}_mergeJob.sub'
	eval 'echo "+JobFlavour			= \"workday\""																		>> test/Jobs/Run${1}_mergeJob.sub'
	eval 'echo "queue"																									>> test/Jobs/Run${1}_mergeJob.sub'

	eval 'echo "executable			= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/testFiles.sh"						>> test/Jobs/Run${1}_testJob.sub'
	eval 'echo "arguments  			= ${1} $CMSSW_BASE ${maxEvents}"													>> test/Jobs/Run${1}_testJob.sub'
	eval 'echo "output 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/LogFiles/Run${1}_testJob.out"	>> test/Jobs/Run${1}_testJob.sub'
	eval 'echo "error 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/LogFiles/Run${1}_testJob.err"	>> test/Jobs/Run${1}_testJob.sub'
	eval 'echo "log 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/LogFiles/Run${1}_testJob.log"	>> test/Jobs/Run${1}_testJob.sub'
	eval 'echo "notify_user			= andrea.bellora@cern.ch"															>> test/Jobs/Run${1}_testJob.sub'
	eval 'echo "notification		= Always"																			>> test/Jobs/Run${1}_testJob.sub'
	eval 'echo "use_x509userproxy   = true"																				>> test/Jobs/Run${1}_testJob.sub'
	eval 'echo "+JobFlavour			= \"nextweek\""																		>> test/Jobs/Run${1}_testJob.sub'
	eval 'echo "queue"																									>> test/Jobs/Run${1}_testJob.sub'

	# eval 'echo "JOB mergeJob test/Jobs/Run${1}_mergeJob.sub"															>> test/Jobs/Run${1}.dag'
	eval 'echo "JOB testJob test/Jobs/Run${1}_testJob.sub"																>> test/Jobs/Run${1}.dag'

	i=0
	while [ $[i*eventsPerJob] -lt $maxEvents ]
	do
		eval 'echo "executable			= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/ReReco.sh"							>> test/Jobs/Run${1}_${i}.sub'
		eval 'echo "arguments  			= ${1} $CMSSW_BASE" $((i*eventsPerJob)) ${eventsPerJob} ${i}						>> test/Jobs/Run${1}_${i}.sub'
		eval 'echo "output 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/LogFiles/Run${1}_${i}.out" 	>> test/Jobs/Run${1}_${i}.sub'
		eval 'echo "error 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/LogFiles/Run${1}_${i}.err" 	>> test/Jobs/Run${1}_${i}.sub'
		eval 'echo "log 				= $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/LogFiles/Run${1}_${i}.log" 	>> test/Jobs/Run${1}_${i}.sub'
		eval 'echo "notify_user			= andrea.bellora@cern.ch"															>> test/Jobs/Run${1}_${i}.sub'
		eval 'echo "notification		= Always"																			>> test/Jobs/Run${1}_${i}.sub'
		eval 'echo "use_x509userproxy   = true"																				>> test/Jobs/Run${1}_${i}.sub'
		eval 'echo "+JobFlavour			= \"tomorrow\""																		>> test/Jobs/Run${1}_${i}.sub'
		eval 'echo "queue"																									>> test/Jobs/Run${1}_${i}.sub'
		
		eval 'echo "JOB job_${i} test/Jobs/Run${1}_${i}.sub"																>> test/Jobs/Run${1}.dag'
		eval 'echo "PARENT testJob CHILD job_${i}"																			>> test/Jobs/Run${1}.dag'
		eval 'echo "PARENT job_${i} CHILD mergeJob"																			>> test/Jobs/Run${1}.dag'
		eval 'echo "RETRY job_${i} 3"																						>> test/Jobs/Run${1}.dag'

		i=$[$i+1]
	done

	eval 'echo "JOB mergeJob test/Jobs/Run${1}_mergeJob.sub"															>> test/Jobs/Run${1}.dag' # execute this even if other jobs fail

	# eval 'echo "FINAL mergeJob test/Jobs/Run${1}_mergeJob.sub"															>> test/Jobs/Run${1}.dag' # execute this even if other jobs fail

	condor_submit_dag -notification Always "test/Jobs/Run${1}.dag" 
	echo "stdout and stderr files are saved in test/LogFiles/Run${1}.out and test/LogFiles/Run${1}.err"
	echo "At the end of the job the output will be saved in /eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/ReRecoOutputTmp_CMSSW_10_6_2/Run${1}.root"
	echo "The output is linked in test/OutputFiles/Run${1}.root"
	echo ""
	echo "Run at the end of the job: ./submitEfficiencyAnalysis.sh ${1}"
	echo ""
fi
