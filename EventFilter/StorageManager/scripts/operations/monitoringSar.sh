#!/bin/bash
# $Id: monitoringSar.sh,v 1.2 2008/10/01 04:14:29 loizides Exp $

##Let's define things first
SAR_CMD="/usr/bin/sar"
SAR_INT=5
SAR_CNT=600

# Start time of the script
DATE=`date +"%F_%H:%M:%S"`

# Scriptname
ScriptFileName=`basename $0`
# Service/Function name
ServiceName="${ScriptFileName}"
# Where is it?
DIRECTORY=`echo "$0"|sed "s/${ScriptFileName}//"`

##Output options
OUTPUTLOGDIR="/var/log"
OUTPUTFILE=${OUTPUTLOGDIR}/`basename ${ScriptFileName} .sh`.log

echo "${DATE} Running ${ScriptFileName}" > ${OUTPUTFILE}

#FileLock Directory
LockFileDir="/var/lock"

PID=$$
ERROR=6

## Let's check if the LockFile Directory is there
if [ ! -e "${LockFileDir}" ] || [ ! -d "${LockFileDir}" ]  
   then
   echo "$PID ERROR: ${LockFileDir} Does not exist or ist not a directory"
   exit ${ERROR}
fi

## Create lock file and protect to avoid orfan lock files
# Our internal user interruption
USER_INTERRUPT=13
LockFileBaseName="${ScriptFileName}"
LockFile=${LockFileDir}/${LockFileBaseName}-${DATE}-${PID}.lock
echo ${PID} >${LockFile}
if [ $? != 0 ]
   then 
   echo "$PID ERROR: creating lockfile ${LockFile}. Probable no permission"
   exit ${ERROR}
fi 
# to delete the lock file in case of interruption
trap 'DATE=`date +"%F_%H:%M:%S"`;echo "$PID INFO: Finishing ${ServiceName} ${DATE} after running $SECONDS s";killall sar;killall sadc;rm ${LockFile}' ${ERROR} EXIT ${USER_INTERRUPT} 1
#If interrupt call trap before but supply correct message
trap 'echo "$PID INFO: USER INTERRUPTED"; exit ${USER_INTERRUPT}' TERM INT

echo "$PID INFO: Starting ${ServiceName} ${DATE}"

#CheckIfOtherLockFiles () {

## Check if other lock file
#PossibleLockFiles=`ls ${LockFileDir}/${LockFileBaseName}-*.lock|grep -v ${LockFile}|tr '\n' ' '`
PossibleLockFiles=`ls ${LockFileDir}/${LockFileBaseName}-*.lock|grep -v ${LockFile}`
#NumberOfLockFiles=`echo $PossibleLockFiles|wc| awk '{print $2;}'`
if [ ! -z "${PossibleLockFiles}" ]
   then 
      echo "$PID WARNING: Found lock files: ${PossibleLockFiles}"
   # Check if files there, not deleted yet, and corresponding to our process
   for file in ${PossibleLockFiles}
   # Check if files there, not deleted yet
      do if [ -e "${file}" ] 
	 then 
	 processIDToTest=`cat ${file}`
	 # Corresponds to our process
	 FoundProcessMatchesNameAndPID=`ps l -p ${processIDToTest} | grep ${LockFileBaseName}`
	 if [ ! -z "${FoundProcessMatchesNameAndPID}" ]; then 
	     echo "$PID ERROR: Found active ${LockFileBaseName} under PID ${processIDToTest} with lock file ${file}"
	     exit ${ERROR}
	 else
	     echo "$PID WARNING: Deleting lock file ${file}. No process ${LockFileBaseName} running under PID ${processIDToTest}"
	     rm ${file}
	 fi
     fi
   done
fi
#}

#CheckIfOtherLockFiles

## No other lock file let's do something!
## Check Usage
if [ "$#" -gt 1 ] # Test number of arguments to script # (always a good idea).
then   
   echo "$PID Usage: ${ScriptFileName}  \"YYYY_MM_DD YYYY_MM_DD \" "
   exit ${ERROR} 
fi

## Check sar
if [ ! -x "${SAR_CMD}" ]
then
   echo "$PID ERROR: ${SAR_CMD} missing."
   exit ${ERROR}
fi

main ()
{
   while [ 1 ]
   do
      DATE=`date +"%F_%H%M"`
      SAR_OUT="${OUTPUTLOGDIR}/${ServiceName}-${HOSTNAME}-${DATE}.sar"
      echo "$PID INFO: Writing to ${SAR_OUT}"
      `${SAR_CMD} -A -o ${SAR_OUT} $SAR_INT $SAR_CNT`
      sleep 1
   done
}

main
