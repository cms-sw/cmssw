#!/bin/sh
# Launch the scripts to perform creation/submission of jobs and retrieval/publication of outputs

####################################
#//////////////////////////////////#
####################################
##           LOCAL PATHS          ##
## change this for your local dir ##
####################################

## Where to find all the templates and to write all the logs
export LOCALHOME=/analysis/sw/CRAB
## Where to copy all the results
export MainStoreDir=/data1/CrabAnalysis
## Where to create crab jobs
export WorkingDir=/tmp/${USER}
## Leave python path as it is to source in standard (local) area
export python_path=/analysis/sw/CRAB

####################################
#//////////////////////////////////#
####################################

########################################
## Patch to make it work with crontab ##
###########################################
export MYHOME=/analysis/sw/CRAB
# the scritps will source ${MYHOME}/crab.sh
###########################################

[ -e ${LOCALHOME}/lock ] && exit

touch ${LOCALHOME}/lock


# Process all the Flags
cat ${LOCALHOME}/Analysis_cron.cfg | grep -v "#" | grep -v "=" | while read FullList; do

  # Skip empty lines
    if [ "${FullList}" != "" ]; then

    # Flag of this processing
	export Version=`echo ${FullList} | awk -F, '{print $1}' | awk '{print $1}'`
	export AnalyzersList=`echo ${FullList} | awk -F, '{print $3}'`
	export RunsList=`echo ${FullList} | awk -F, '{print $4}'`
## Where to do eval scramv1 (without src/)
	export CMSSW_DIR=${LOCALHOME}/CMSSW/`echo ${FullList} | awk -F, '{print $2}' | sed -e "s@[ \t]*@@g"`
	
    # Check if there is at least one analyzer and run to process
	if [ `echo ${AnalyzersList} | awk '{print $1}'` != "" ] && [ `echo ${RunsList} | awk '{print $1}'` != "" ]; then
	   	    

      # Paths
      #######
      # local dirs
	    export local_crab_path=${LOCALHOME}
	    export cfg_path=${local_crab_path}/cfg/${Version}
	    export template_path=${local_crab_path}/templates/${Version}
	    export log_path=${local_crab_path}/log/${Version}
      # log path
	    export created_path=${log_path}/Created
	    export not_created_path=${log_path}/Not_Created
	    export submitted_path=${log_path}/Submitted
	    export not_submitted_path=${log_path}/Not_Submitted
      # list dirs
	    export list_path=${log_path}/list
	    export list=list_reco_CMSSW_1_3_0_pre6.txt
	    export list_phys=list_physics_runs.txt
#      export list_selected=list_selected_runs.txt

      ############
      ## PYTHON ##
      ############
	    export PYTHONPATH=$PYTHONPATH:${python_path}/COMP/DBS/Clients/PythonAPI
	    export PYTHONPATH=$PYTHONPATH:${python_path}/COMP/DLS/Client/LFCClient
	    export PYTHONPATH=$PYTHONPATH:${python_path}/COMP/DLS/Client/DliClient
	    export PATH=$PATH:${python_path}/COMP/:${python_path}/COMP/DLS/Client/LFCClient
      ############

	    mkdir -p ${MainStoreDir}/logs/${Version}/LogProducer/
	    mkdir -p ${MainStoreDir}/logs/${Version}/LogMonitor/${Version}

	    echo -e "\nExecuting producer: creating and submitting new jobs"
#      if [ ! -e ${LOCALHOME}/lock ]; then
#        touch ${LOCALHOME}/lock

	    echo -e "\n${LOCALHOME}/Producer.sh ${Version} > ${MainStoreDir}/logs/${Version}/LogProducer/prod_log_`date +\%Y-\%m-\%d_\%H-\%M-\%S`"

	    ${LOCALHOME}/Producer.sh ${Version} #> ${MainStoreDir}/logs/${Version}/LogProducer/prod_log_`date +\%Y-\%m-\%d_\%H-\%M-\%S`
#        rm -f ${LOCALHOME}/lock
#      fi

	    echo -e "\nExecuting monitor: checking jobs status, retrieving and publishing output"
#      if [ ! -e ${LOCALHOME}/lock ]; then
#        touch ${LOCALHOME}/lock

	    echo -e "\n${LOCALHOME}/Monitor.sh ${Version} > ${MainStoreDir}/logs/${Version}/LogMonitor/${Version}/monitor_log_`date +\%Y-\%m-\%d_\%H-\%M-\%S`"

	    ${LOCALHOME}/Monitor.sh ${Version} > ${MainStoreDir}/logs/${Version}/LogMonitor/monitor_log_`date +\%Y-\%m-\%d_\%H-\%M-\%S`

#        rm -f ${LOCALHOME}/lock
#      fi

	    if [ ! -e ${LOCALHOME} ]; then
		echo FUNCTIONING_3
	    fi
	else
	    if [ `echo ${AnalyzersList} | awk '{print $1}'` == "" ]; then
		echo Specify at least one Analyzer name
	    elif [ `echo ${RunsList} | awk '{print $1}'` == "" ]; then
		echo Specify at least one Run number or use All to process all runs
	    else
		echo Something strange happened
	    fi
	fi
    fi
done

echo -e "\n...Creating Summaries"
${LOCALHOME}/getSummary.sh
	    
echo -e "\n...Running BadStripsFromPosition"
${LOCALHOME}/macros/BadStripsFromPosition.sh


rm -f ${LOCALHOME}/lock