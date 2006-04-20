# A script to run as a cron job to execute O2O for EcalPedestals
# Prerequisites: 1. A MV group defined on the offline DB
#                2. A bootstrapped CMSSW project to set your
#                   environment to use the programs setup_pool_database and
#                   cmscond_build_iov 
#                3. The mapping xml files for the object to do o2o on
# Warning: O2O needs to have been executed once before, or remove
#          the -a option on cmscond_build_iov for the first execution
# TODO:  CORAL_AUTH_* shouldn't need to be set!  Should use authentication XML

# Basic paths and variables
O2ODIR=$HOME/scratch0
CMSSW_VER=CMSSW_0_6_0_pre1
SCRAM_PATH=/afs/cern.ch/cms/utils
SCRAM_ARCH=slc3_ia32_gcc323
CMSSW_DIR=${O2ODIR}/${CMSSW_VER}
LOG=$O2ODIR/o2o-log.txt

# Info about the object to be o2o'd
MAPPING_PATH=${CMSSW_DIR}/src/CondTools/Ecal/src
MAPPING_FILE=${MAPPING_PATH}/EcalPedestals-mapping-custom_1.0.xml
OBJECT_LIBRARY=CondFormatsEcalObjects
OBJECT_NAME=EcalPedestals
OBJECT_TABLE=ECALPEDESTALS
TAG=from_online

# Online DB info
ONLINE_DB=cmsomds
ONLINE_DB_USER=cms_ecal
ONLINE_DB_PASSWORD=*********

# Offline DB info
OFFLINE_DB=orcon
OFFLINE_DB_USER=CMS_COND_ECAL # uppercase for connect string
OFFLINE_DB_PASSWORD=*******
OFFLINE_CONNECT=oracle://${OFFLINE_DB}/${OFFLINE_DB_USER}
MVGRP=ECALGRP

# POOL catalog info
#MY_CATALOG=xmlcatalog_file:${O2ODIR}/orcon-test-catalog.xml
MY_CATALOG=relationalcatalog_oracle://orcon/cms_cond_general

# Log the date
echo -n [`date "+%Y-%m-%d %H:%M:%S"`] >> $LOG;
T_START=`date +%s`


# Set the environment
echo -n " Setting env..." >> $LOG;
PATH=$PATH:$SCRAM_PATH
cd $CMSSW_DIR;
eval `scramv1 runtime -sh`;
cd $O20DIR;

COND_UTIL_PATH=${LOCALRT}/src/CondTools/Utilities/bin
PATH=$PATH:$COND_UTIL_PATH

export POOL_CATALOG=${MY_CATALOG}
echo $POOL_CATALOG
export CORAL_AUTH_USER=$OFFLINE_DB_USER
export CORAL_AUTH_PASSWORD=$OFFLINE_DB_PASSWORD
#export POOL_AUTH_PATH=${O2ODIR}

###
### O2O happens here
###

# Transform payload data
T1=`date +%s`
echo -n "Updating payload tables..." >> $LOG;
#echo "exec dbms_refresh.refresh('${MVGRP}');" | sqlplus -S ${OFFLINE_DB_USER}/${OFFLINE_DB_PASSWORD}@${OFFLINE_DB} 2>> $LOG
echo "call payload_o2o();" | sqlplus -S ${OFFLINE_DB_USER}/${OFFLINE_DB_PASSWORD}@${OFFLINE_DB} 2>> $LOG
T2=`date +%s`
T_JOB=$(($T2-$T1))
echo -n "($T_JOB s)" >> $LOG;


# Poolify offline objects
T1=`date +%s`
echo -n "Registering to POOL..." >> $LOG;
setup_pool_database $OBJECT_NAME \
                    $OBJECT_LIBRARY \
                    $OFFLINE_CONNECT \
                    $MAPPING_FILE -o $O2ODIR 2>>$LOG
T2=`date +%s`
T_JOB=$(($T2-$T1))
echo -n "($T_JOB s)" >> $LOG;

# Assign iov
T1=`date +%s`
echo -n "Assigning IOV..." >> $LOG;
cmscond_build_iov -c $OFFLINE_CONNECT \
                  -d $OBJECT_LIBRARY \
                  -t $OBJECT_TABLE \
                  -o $OBJECT_NAME \
                  -a $TAG 2>>$LOG
T2=`date +%s`
T_JOB=$(($T2-$T1))
echo -n "($T_JOB s)" >> $LOG;


# Log the duration of the O2O
T_FINISH=`date +%s`;
T_JOB=$(($T_FINISH-$T_START))
echo "Done ($T_JOB s)." >> $LOG;
