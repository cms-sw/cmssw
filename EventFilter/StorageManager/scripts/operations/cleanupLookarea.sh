#!/bin/sh

#FileLock Directory
#LockFileDir="/var/lock"
LockFileDir="/tmp/"

#script to clean up LOOKAREA --- to run on cms-data-lookarea
#(derived from  cleanupEmu.sh)


LOCALNODE=`hostname | tr '[A-Z]' '[a-z]'` 

ISLOOKNODE=`host cms-data-lookarea | grep -ic $LOCALNODE`
if  test "$ISLOOKNODE" -eq  "0"; then
    echo "  "
    echo "        **** WRONG NODE: $LOCALNODE is not cms-data-lookarea ***  "
    echo "  "
    exit 12
fi



# create/check lock file
PID=$$
if [ ! -e "${LockFileDir}" ] || [ ! -d "${LockFileDir}" ]
   then
   echo "ERROR: ${LockFileDir} Does not exist or is not a directory"
   exit ${ERROR}
fi
ER_INTERRUPT=13
#LockFileBaseName="$0"
LockFileBaseName=`echo $0 | awk -F/ '{print $NF}' `



DATE=`date +"%F_%H:%M:%S"`
LockFile=${LockFileDir}/${LockFileBaseName}-${DATE}-${PID}.lock
echo ${PID} >${LockFile}
if [ $? != 0 ]
   then
   echo "ERROR: creating lockfile ${LockFile}. Probable no permission"
   exit ${ERROR}
fi
trap 'DATE=`date +"%F_%H:%M:%S"`;echo "INFO: Finishing ${0} ${DATE}";rm ${LockFile}' EXIT 1
trap 'echo "ERROR: USER INTERRUPTED"; exit -1' TERM INT
echo "INFO: Starting ${DATE}"
PossibleLockFiles=`ls ${LockFileDir}/${LockFileBaseName}-*.lock|grep -v ${LockFile}`
if [ ! -z "${PossibleLockFiles}" ]
   then
       echo "WARNING: Found lock files: ${PossibleLockFiles}"
       for file in ${PossibleLockFiles}; do
           if [ -e "${file}" ]; then
               processIDToTest=`cat ${file}`
               FoundProcessMatchesNameAndPID=`ps l -p ${processIDToTest} | grep ${LockFileBaseName}`
               if [ ! -z "${FoundProcessMatchesNameAndPID}" ]; then
                   echo "ERROR: Found active ${LockFileBaseName} under PID ${processIDToTest} with lock file ${file}"
                   exit -1
               else
                   echo "WARNING: Deleting lock file ${file}. No process ${LockFileBaseName} running under PID ${processIDToTest}"
                   rm ${file}
               fi
           fi
       done
fi

#Path of LOOKAREA---directory to cleanup:
LOOKDIR="/lookarea_SM"
if ! test -d "$LOOKDIR"; then
    echo "Dir $LOOKDIR not found or not a directory"
    exit 123
fi

CUDdir="$LOOKDIR"
mntpoint="$LOOKDIR"


#Delete BIG/SMALL files with different ages,
# cut-off size between BIG and SMALL files, in kilobytes:
SMALLFILESIZE=35840

#convert to MB for human convenience:
SMALLFILESIZE_M=$(echo "scale=2;$SMALLFILESIZE/1024 " | bc) 



#deletion cutoffs:
# lifetime in mins / 1 day = 1440
LIFETIME_SMLL90=120     #  2 hrs
LIFETIME_SMLL82=360     #  6 hrs
LIFETIME_SMLL73=720     # 12 hrs 
LIFETIME_SMLL65=1440    # 1 day
LIFETIME_SMLL55=4320    # 3 days
LIFETIME_SMLL35=7200    # 5 days 
LIFETIME_SMLL25=10800   # 7 days
LIFETIME_SMLL13=20160   #14 days
LIFETIME_SMLL00=50400   #35 days



LIFETIME_BIG85=60      #  1 hr
LIFETIME_BIG80=180     #  3 hr
LIFETIME_BIG75=360     #  6 hrs
LIFETIME_BIG70=720     # 12 hrs 
LIFETIME_BIG60=1440    # 1 day
LIFETIME_BIG50=2880    # 2 days
LIFETIME_BIG40=4320    # 3 days
LIFETIME_BIG30=7200    # 5 days 
LIFETIME_BIG20=10080   # 7 days
LIFETIME_BIG10=20160   #14 days
LIFETIME_BIG00=50400   #35 days





# delete small files first:

    # find how full disk is to determine how much to delete
LIFETIME_SMLL=$(df | 
    awk -v LIFETIME_SMLL90="$LIFETIME_SMLL90" \
	-v LIFETIME_SMLL82="$LIFETIME_SMLL82" \
	-v LIFETIME_SMLL73="$LIFETIME_SMLL73" \
	-v LIFETIME_SMLL65="$LIFETIME_SMLL65" \
	-v LIFETIME_SMLL55="$LIFETIME_SMLL55" \
	-v LIFETIME_SMLL35="$LIFETIME_SMLL35" \
	-v LIFETIME_SMLL25="$LIFETIME_SMLL25" \
	-v LIFETIME_SMLL13="$LIFETIME_SMLL13" \
	-v LIFETIME_SMLL00="$LIFETIME_SMLL00" \
	-v pat="$mntpoint" \
	'$0 ~ pat {if ($4+0 > 90) print LIFETIME_SMLL90; \
	else if ($4+0 > 82) print LIFETIME_SMLL82; \
	else if ($4+0 > 73) print LIFETIME_SMLL73; \
	else if ($4+0 > 65) print LIFETIME_SMLL65; \
	else if ($4+0 > 55) print LIFETIME_SMLL55; \
	else if ($4+0 > 35) print LIFETIME_SMLL35; \
	else if ($4+0 > 25) print LIFETIME_SMLL25; \
	else if ($4+0 > 13) print LIFETIME_SMLL13; \
	else print LIFETIME_SMLL00; }' )


    #clean SMALL files

#increment threshold cuz tests are only > or <, not =
SMALLFILESIZE=$(echo "scale=0;$SMALLFILESIZE+1 " | bc) 



NDELETEDSMLL=`find $CUDdir -cmin +$LIFETIME_SMLL -type f -a -size -"$SMALLFILESIZE"k  -a  -exec rm -fv {}  \; | grep -c removed`





DELETETIME_hr=$(echo "scale=2;$LIFETIME_SMLL/60 " | bc) 
DELETETIME_day=$(echo "scale=1;$DELETETIME_hr/24 " | bc) 
echo " >>>> Deleted $NDELETEDSMLL files older than $LIFETIME_SMLL ($DELETETIME_hr hrs; or $DELETETIME_day days) and smaller than  $SMALLFILESIZE kB ($SMALLFILESIZE_M MB)"





# then delete big files:

    # find how full disk is to determine how much to delete
LIFETIME_BIG=$(df | 
    awk -v LIFETIME_BIG85="$LIFETIME_BIG85" \
	-v LIFETIME_BIG80="$LIFETIME_BIG80" \
	-v LIFETIME_BIG75="$LIFETIME_BIG75" \
	-v LIFETIME_BIG70="$LIFETIME_BIG70" \
	-v LIFETIME_BIG60="$LIFETIME_BIG60" \
	-v LIFETIME_BIG50="$LIFETIME_BIG50" \
	-v LIFETIME_BIG40="$LIFETIME_BIG40" \
	-v LIFETIME_BIG30="$LIFETIME_BIG30" \
	-v LIFETIME_BIG20="$LIFETIME_BIG20" \
	-v LIFETIME_BIG10="$LIFETIME_BIG10" \
	-v LIFETIME_BIG00="$LIFETIME_BIG00" \
	-v pat="$mntpoint"  \
	'$0 ~ pat {if ($4+0 > 85) print LIFETIME_BIG85; \
	else if ($4+0 > 80) print LIFETIME_BIG80; \
	else if ($4+0 > 75) print LIFETIME_BIG75; \
	else if ($4+0 > 70) print LIFETIME_BIG70; \
	else if ($4+0 > 60) print LIFETIME_BIG60; \
	else if ($4+0 > 50) print LIFETIME_BIG50; \
	else if ($4+0 > 40) print LIFETIME_BIG40; \
	else if ($4+0 > 30) print LIFETIME_BIG30; \
	else if ($4+0 > 20) print LIFETIME_BIG20; \
	else if ($4+0 > 10) print LIFETIME_BIG10; \
	else print LIFETIME_BIG00; }' )


    #clean BIG
###CUDdir="$LOOKDIR"
NDELETEDBIG=`find $CUDdir -cmin +$LIFETIME_BIG -type f -a -size +"$SMALLFILESIZE"k  -a  -exec rm -fv {} \; | grep -c removed`



DELETETIME_hr=$(echo "scale=2;$LIFETIME_BIG/60 " | bc) 
DELETETIME_day=$(echo "scale=1;$DELETETIME_hr/24 " | bc) 

echo " >>>> Deleted $NDELETEDBIG files older than $LIFETIME_BIG ($DELETETIME_hr hrs; or $DELETETIME_day days) and larger than $SMALLFILESIZE kB ($SMALLFILESIZE_M MB)"



exit 0;


