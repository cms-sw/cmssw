#!/bin/bash



# Process arguments and set the flags
fileName=$1

# Obtain the runList from a file, if needed
runList=""
if [ ${#fileName} -gt 0 ] ; then
  if [ -s ${fileName} ] ; then
      runList=`cat ${fileName}`
  else
      echo "<${fileName}> does not seem to be a valid file"
      exit 2
  fi
else
    if [ ${ignoreFile} -eq 0 ] ; then
	echo " ! no file provided"
    fi
    echo " ! will produce only the global html page"
fi

# create log directory
LOG_DIR="dir-Logs"
if [ ! -d ${LOG_DIR} ] ; then mkdir ${LOG_DIR}; fi
rm -f ${LOG_DIR}/*

#processing

for i in ${runList} ; do
    runnumber=$i

    logFile="${LOG_DIR}/log_${runnumber}.out"
    rm -f ${logFile}

    #GlobalPSM processing
    echo -e "\nRemoteMonitoringPSM\n" >> ${logFile}
    ./../../macros/psm/RemoteMonitoringPSM.cc.exe GlobalPSM_$runnumber.root 2>&1 | tee -a ${logFile}
    if [ ! $? -eq 0 ] ; then
	echo "PSM processing failed"
	exit 2
    fi

    local_WebDir=dir-CMT-GLOBAL_${runnumber}
    rm -rf ${local_WebDir}
    if [ ! -d ${local_WebDir} ] ; then mkdir ${local_WebDir}; fi
    for j in $(ls -r *.html); do
	cat $j | sed 's#cms-cpt-software.web.cern.ch\/cms-cpt-software\/General\/Validation\/SVSuite#cms-conddb.cern.ch\/eosweb\/hcal#g' \
	    > ${local_WebDir}/$j
    done
    mv *.png ${local_WebDir}
    mv *.html ${local_WebDir}
##    cp HELP.html ${local_WebDir}
#   removing:
#
#
#
    rm *.html
    rm *.png 
#    rm -rf dir-CMT-GLOBAL_*
#    rm *.root
#
done

#---------------

#fi



echo "CMT script done"
#---------------
