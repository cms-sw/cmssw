# Basic paths, files, and variables
O2ODIR=$HOME/scratch0
CMSSW_VER=CMSSW_0_7_0_pre5
SCRAM_PATH=/afs/cern.ch/cms/utils
SCRAM_ARCH=slc3_ia32_gcc323
CMSSW_DIR=${O2ODIR}/${CMSSW_VER}
if [ ! -d $CMSSW_DIR ] 
then
    echo "$CMSSW_DIR does not exist.  Please check O2O_DIR and CMSSW_VER in general-runtime.sh"
    exit
fi

LOG=$O2ODIR/o2o-log.txt

# General object setup
MAPPING_PATH=${CMSSW_DIR}/src/CondTools/IntegrationTest/mappings

# Path to SQL scripts
SQL_PATH=${CMSSW_DIR}/src/CondTools/IntegrationTest/sql

# Set the CMSSW environment
PATH=$PATH:$SCRAM_PATH
CURR_DIR=`pwd`
cd $CMSSW_DIR;
eval `scramv1 runtime -sh`;
COND_UTIL_PATH=${LOCALRT}/src/CondTools/Utilities/bin
PATH=$PATH:$COND_UTIL_PATH
cd $CURR_DIR

# Get the general DB setup
source general-db-setup.sh
