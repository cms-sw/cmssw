#! /bin/bash
#Prerequesits: 
#   -CMS and LCG software are installed on the machine
#   -The data payload are already written in the database, IOV is assigned to the payload and the IOV is tagged. This script assumes your tag is ${detector}fall_test where ${detector} is your detector name in lowercase
#   -In case of oracle database, a preallocated oracle catalog is used; in case of sqlite database, the datafile and the catalog file, PoolFileCatalog.xml,generated when writing the data should ALWAYS be moved around together
#INSTRUCTION:
#   -mkdir ${workingdir}
#   -bootstrap ${CMSSWVERSION} project in the workingdir
#   -cd ${CMSSWVERSION}; download this script; 
#   -Change the setup environment section according to the parameters you use for the test
#    -chmod a+x condReaderTest.sh
#    -./condReaderTest.sh
#    This script runs the full chain from boostraping CMSSW, generating the configuration file to run the test
#---------------------------------------------------------------------
# setup environment and user parameters
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
THISDIR=`pwd`
export SCRAM_ARCH=slc3_ia32_gcc323
CMSSWVERSION=CMSSW_2006-02-24 #change!!!
OWNER=ECAL #change!!!
OWNERPASS=cern2006x #change!!!
MAXEVENTS=10
FIRSTRUN=1
EVENTSINRUN=1
SERVICENAME=devdb10
SERVERNAME=devdb10
SERVERHOSTNAME=oradev10.cern.ch
OWNERSCHEMA=CMS_COND_${OWNER}
GENSCHEMA=CMS_COND_GENERAL
LOCALCATALOG=conddbcatalog.xml
# ------------------------------------------------------------------------
# setup_tns ()
# write tnsnames.ora in the current directory and set variable TNS_ADMIN for running this job to this directory
# Parameters:
# Returns; 0 on success
# ------------------------------------------------------------------------
setup_tns() {
local TNSFILE=tnsnames.ora
rm -f ${TNSFILE}
/bin/cat >  ${TNSFILE} <<EOI
devdb10=(DESCRIPTION=
        (ADDRESS=
                (PROTOCOL=TCP)
                (HOST=oradev10.cern.ch)
                (PORT=10520)
        )
        (CONNECT_DATA=
                (SID=D10))
        )
EOI
#uncomment if your working node doesnot recognise devdb10
#export TNS_ADMIN=${THISDIR}
#echo "[---JOB LOG---] Using TNS_ADMIN=${TNS_ADMIN}, ORACLE server ${SERVERNAME} at host ${SERVERHOSTNAME}"
return 0
}
#-------------------------------------------------------------------------
# bootstrap_cmssw ()
# bootstrap a new CMSSW project, removing old if exists
# Parameters: CMSSWVERSION($1)
# Returns; 0 on success
#-------------------------------------------------------------------------
bootstrap_cmssw () {
if [ -d $1 ]; 
    then /bin/rm -rf $1 ;
fi 
scramv1 project CMSSW $1
cd $1/src
eval `scramv1 runtime -sh`
cd ${THISDIR}
return 0
}
# ------------------------------------------------------------------------
# write_config ()
# write the job configuration file
# Parameters: OWNER($1), OBJNAME($2), MAXEVENTS($3), FIRSTRUN($4), EVENTSINRUN($5)
# Returns; 0 on success
# ------------------------------------------------------------------------
write_config() {
CONFFILE=condRead$2.cfg
local CONNECT=oracle://${SERVERNAME}/${OWNERSCHEMA}
local RCD=$2Rcd
#local TAG=`echo $1| awk '{print tolower($1)}'`_test 
local TAG=ecal_test #change!!!
/bin/cat >  ${CONFFILE} <<EOI
  process condTEST = {
	path p = { get }

	es_source = PoolDBESSource { VPSet toGet = {
                                   {string record = "${RCD}"
                                     string tag = "${TAG}"
                                    } }
		    		    bool loadAll = true
                                    string connect = "${CONNECT}"
                                    untracked string catalog = "file:${LOCALCATALOG}"
			            string timetype = "runnumber" 
                                    untracked uint32 authenticationMethod = 0
				   }
	
	source = EmptySource {untracked int32 maxEvents = $3 
                untracked uint32 firstRun = $4 
                untracked uint32 numberEventsInRun = $5}

	module get = EventSetupRecordDataGetter { VPSet toGet = {
	       {string record = "${RCD}"
	        vstring data = {"$2"} } 
	       }
	       untracked bool verbose = true 
	}
}
EOI
return 0
}

#main
#bootstrap_cmssw ${CMSSWVERSION}
#echo "[---JOB LOG---] bootstrap_cmssw status $?"
#
#uncomment this if your working node doesnot recognise devdb10
#
#setup_tns
#echo  "[---JOB LOG---] setup_tns status $?"
export CORAL_AUTH_USER=${OWNERSCHEMA}
echo ${OWNERSCHEMA}
export CORAL_AUTH_PASSWORD=${OWNERPASS}
echo ${OWNERPASS}
rm -f ${THISDIR}/${LOCALCATALOG}
echo "[---JOB LOG---] Publishing catalog"
echo "FCpublish -u relationalcatalog_oracle://${SERVERNAME}/${GENSCHEMA} -d file:${LOCALCATALOG}"
FCpublish -u relationalcatalog_oracle://${SERVERNAME}/${GENSCHEMA} -d file:${LOCALCATALOG}
echo "[---JOB LOG---] done"
export POOL_CATALOG=file:${THISDIR}/${LOCALCATALOG}
for PARAM in "ECAL EcalPedestals" ; do
  set -- $PARAM
  write_config $1 $2 ${MAXEVENTS} ${FIRSTRUN} ${EVENTSINRUN}
  echo "[---JOB LOG---] write_config $1 status $?"
  echo "[---JOB LOG---] running job for $1 $2 using ${CONFFILE}" 
  time cmsRun --parameter-set ${CONFFILE}
done
echo "[---JOB LOG---] done"
exit 0


