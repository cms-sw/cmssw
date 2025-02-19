#! /bin/bash
source stats.sh
source setup.sh
#Prerequesits: 
#   -CMS and LCG software are installed on the machine
#   -The data payload are already written in the database, IOV is assigned to the payload and the IOV is tagged. This script assumes your tag is ${detector}fall_test where ${detector} is your detector name in lowercase
#   -In case of oracle database, a preallocated oracle catalog is used; in case of sqlite database, the datafile and the catalog file, PoolFileCatalog.xml,generated when writing the data should ALWAYS be moved around together
#INSTRUCTION:
#   -mkdir ${workingdir}
#   -cd ${workingdir}; download this script; 
#   -Change the setup environment section according to the parameters you use for the test
#    -chmod a+x condReaderTest.sh
#    -./condReaderTest.sh
#    This script runs the full chain from boostraping CMSSW, generating the configuration file to run the test
#---------------------------------------------------------------------
# setup environment and user parameters
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
THISDIR=`pwd`
CMSSWVERSION=CMSSW_0_2_0_pre7 #change to the CMSSW version for testing
export CVSROOT=:kserver:cmscvs.cern.ch:/cvs_server/repositories/CMSSW
export SCRAM_ARCH=slc3_ia32_gcc323
MAXEVENTS=10000
FIRSTRUN=1
EVENTSINRUN=1
SERVICENAME=cms_val_lb
SERVERNAME=${SERVICENAME}.cern.ch
SERVERHOSTNAME=int2r1-v.cern.ch
GENREADER=CMS_VAL_GENERAL_POOL_READER
GENREADERPASS=val_gen_rea_1031
GENSCHEMA=CMS_VAL_GENERAL_POOL_OWNER
NUMTESTS=5
#main
bootstrap_cmssw ${CMSSWVERSION}
echo "[---JOB LOG---] bootstrap_cmssw status $?"
setup_tns
echo  "[---JOB LOG---] setup_tns status $?"
export POOL_AUTH_USER=${GENREADER}
export POOL_AUTH_PASSWORD=${GENREADERPASS}
rm -f ${THISDIR}/conddbcatalog.xml
echo "[---JOB LOG---] Publishing catalog"
FCpublish -u relationalcatalog_oracle://${SERVERNAME}/${GENSCHEMA} -d file:conddbcatalog.xml
export POOL_CATALOG=file:${THISDIR}/conddbcatalog.xml
#export POOL_OUTMSG_LEVEL=8

for PARAM in "ECAL EcalPedestals"; do
  set -- $PARAM
  echo "[---JOB LOG---] Writing parameter-set file:  MAXEVENTS=$MAXEVENTS FIRSTRUN=$FIRSTRUN EVENTSINRUN=$EVENTSINRUN"
  write_config $1 $2 ${MAXEVENTS} ${FIRSTRUN} ${EVENTSINRUN}
  echo "[---JOB LOG---] write_config $1 status $?"
  echo "[---JOB LOG---] Running job for $1 $2 using ${CONFFILE}" 
  runx "cmsRun --parameter-set ${CONFFILE}" $NUMTESTS
done
exit 0


