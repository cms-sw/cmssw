# A script to execute O2O for a given subdetector and POOL-ORA object

# Check arguments
if [ ! -n "$3" ]  
then
  echo "Usage: `basename $0` <subdetector> <object> <append>"
  echo "       subdetector:  Name of subdetector (ECAL, HCAL, CSC, etc.)"
  echo "       object:  Name of POOL-ORA object to execute O2O for"
  echo "       append:  boolean value, if true then append IOV mode is on"
  exit -1
fi

SUBDETECTOR=$1
OBJECT=$2
APPEND=$3

SUBDETECTOR_SETUP=${SUBDETECTOR}-db-setup.sh
OBJECT_SETUP=${OBJECT}-object-setup.sh

for file in $SUBDETECTOR_SETUP $OBJECT_SETUP
do
  if [ ! -f "$file" ]
    then
      echo "ERROR:  $file not found" >&2
      exit -1
  fi
done

if [ $APPEND = 1 ]
then
  APPEND="-a"
else
  APPEND=
fi


# Basic paths, files, and variables
O2ODIR=$HOME/scratch0
CMSSW_VER=CMSSW_0_6_0_pre4
SCRAM_PATH=/afs/cern.ch/cms/utils
SCRAM_ARCH=slc3_ia32_gcc323
CMSSW_DIR=${O2ODIR}/${CMSSW_VER}
LOG=$O2ODIR/o2o-log.txt

# General DB setup
OFFLINE_DB=orcon
GENERAL_DB_USER=CMS_COND_GENERAL
GENERAL_DB_PASSWORD=******
MY_CATALOG=relationalcatalog_oracle://${OFFLINE_DB}/${GENERAL_DB_USER}

# Subdetector-specific DB setup
# Sets SUBDETECTOR_DB_USER SUBDETECTOR_DB_PASSWORD
source $SUBDETECTOR_SETUP
OFFLINE_CONNECT=oracle://${OFFLINE_DB}/${SUBDETECTOR_DB_USER}

# General object setup
MAPPING_PATH=${CMSSW_DIR}/src/CondTools/IntegrationTest/mappings

# Subdetector-specific object setup
# Sets MAPPING_FILE OBJECT_LIBRARY OBJECT_NAME OBJECT_TABLE TAG
source $OBJECT_SETUP

# Log the date
echo -n [`date "+%Y-%m-%d %H:%M:%S"`] >> $LOG;
T_START=`date +%s`


# Set the CMSSW environment
echo -n " Setting env..." >> $LOG;
PATH=$PATH:$SCRAM_PATH
cd $CMSSW_DIR;
eval `scramv1 runtime -sh`;
cd $O20DIR;

COND_UTIL_PATH=${LOCALRT}/src/CondTools/Utilities/bin
PATH=$PATH:$COND_UTIL_PATH

export POOL_CATALOG=${MY_CATALOG}
export CORAL_AUTH_USER=$SUBDETECTOR_DB_USER
export CORAL_AUTH_PASSWORD=$SUBDETECTOR_DB_PASSWORD

# Additional setup checks
if [ ! -f "$MAPPING_FILE" ]
then
  echo "ERROR:  Mapping file $MAPPING_FILE not found" >&2
  exit -1
fi

###
### O2O happens here
###

# Transform payload data
T1=`date +%s`
echo -n "Updating payload tables..." >> $LOG;
echo "call master_payload_o2o('${OBJECT_NAME}');" | sqlplus -S ${GENERAL_DB_USER}/${GENERAL_DB_PASSWORD}@${OFFLINE_DB} 2>> $LOG
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
                  $APPEND $TAG 2>>$LOG
T2=`date +%s`
T_JOB=$(($T2-$T1))
echo -n "($T_JOB s)" >> $LOG;

# Log the duration of the O2O
T_FINISH=`date +%s`;
T_JOB=$(($T_FINISH-$T_START))
echo "Done ($T_JOB s)." >> $LOG;

tail -n1 $LOG
