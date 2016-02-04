#!/bin/bash

#  ./run_analysis.sh 
#    1: <run_number>
#    2: <HWUPLOAD>
#    3: <AnalysisUpload>
#    4: <DB_PARTITION>
#    5: <UseClientFile>
#    6: <DisableDevices>
#    7: <saveClientFile>

# tests on env
if [ -n "`uname -n | grep vmepc`" -a `whoami` == "trackerpro" ] ; then
    :
elif [ -n "`uname -n | grep cmstracker029`" -a `whoami` == "xdaqtk" ] ; then
    :
else
    echo "You are not running as trackerpro (on vmepcs) or xdaqtk (on cmstracker029).";
    echo "This can cause problems during file moving, like loss of raw data.";
    echo "You don't want that to happen, probably, so please login as the appropriate user and try again.";
    exit 0;
fi


# tests for proper usage
function usage() {
    echo "Usage:  ./run_analysis.sh <run_number> <HWUPLOAD> <AnalysisUpload> <DB_partition_name> <UseClientFile> <DisableDevices> <saveClientFile>"
    echo "  run_number        = run number"
    echo "  HWUpload          = set to true if you want to upload the HW config to the DB"
    echo "  AnalysisUpload    = set to true if you want to upload the analysis results to the DB"
    echo "  DB_partition_name = Name of the corresponding DB partition"
    echo "  UseClientFile     = set to true if you want to analyze an existing client file rather than the source file(s)"
    echo "  DisableDevices    = set to true if you want to disable devices in the DB (normally set False)"
    echo "  saveClientFile    = set to true if you want to write the client file to disk (normally set True)"
}
if [ $# -eq 0 ]; then
    usage
    exit 0
fi
if [ $1 = "usage" ]; then
    usage
    exit 0
fi
if [ $# -lt 7 ]; then
    echo "Not enough arguments specified!"
    usage
    exit 0
fi

echo "Running analysis script ..."

# Parse command line parameters
RUNNUMBER=$1
HWUPLOAD=$2         ; HWUPLOAD=`echo $HWUPLOAD | tr 'ft' 'FT'`
ANALYSISUPLOAD=$3   ; ANALYSISUPLOAD=`echo $ANALYSISUPLOAD | tr 'ft' 'FT'`
DBPARTITIONNAME=$4
USECLIENTFILE=$5    ; USECLIENTFILE=`echo $USECLIENTFILE | tr 'ft' 'FT'`
DISABLEDEVICES=$6   ; DISABLEDEVICES=`echo $DISABLEDEVICES | tr 'ft' 'FT'`
DISABLEBADSTRIPS="False"
SAVECLIENTFILE=$7   ; SAVECLIENTFILE=`echo $SAVECLIENTFILE | tr 'ft' 'FT'`

# Settings for basic directories
BASEDIR=/opt/cmssw
echo "  CMSSW base directory     : "$BASEDIR
DATALOC=/opt/cmssw/Data
echo "  Temporary storage area   : "$DATALOC
SCRATCH=$BASEDIR/Data/$RUNNUMBER
echo "  Output storage directory : "$SCRATCH
TEMPLATEPY=/opt/cmssw/scripts/analysis_template_cfg.py
echo "  Analysis template        : "$TEMPLATEPY
echo "  ConfDB account           : "$CONFDB

# set up CMSSW environment
source $BASEDIR/scripts/setup.sh
cd $BASEDIR/Stable/current/src
eval `scram runtime -sh`

# make the output storage directory if it does not already exist
if [ ! -d $SCRATCH ]; then
  mkdir -p $SCRATCH
fi

# copy over the source file
for SOURCEFILE in `ls $DATALOC | grep \`printf %08u $RUNNUMBER\` | grep Source`; do
  mv $DATALOC/$SOURCEFILE $SCRATCH
done

# make the analysis config file to run
sed 's,RUNNUMBER,'$RUNNUMBER',g' $TEMPLATEPY \
  | sed 's,DBUPDATE,'$HWUPLOAD',g' \
  | sed 's,ANALUPDATE,'$ANALYSISUPLOAD',g' \
  | sed 's,DBPART,'$DBPARTITIONNAME',g' \
  | sed 's,CLIENTFLAG,'$USECLIENTFILE',g' \
  | sed 's,DATALOCATION,'$SCRATCH',g' \
  | sed 's,DISABLEDEVICES,'$DISABLEDEVICES',g' \
  | sed 's,DISABLEBADSTRIPS,'$DISABLEBADSTRIPS',g' \
  | sed 's,SAVECLIENTFILE,'$SAVECLIENTFILE',g' \
  > $SCRATCH/analysis_${RUNNUMBER}_cfg.py

# run the analysis!
source /opt/trackerDAQ/config/oracle.env.bash
cd $SCRATCH
DATESUFF=`date +%s`  # to get consistent dates across logfiles
echo -n "  Running analysis ... "
cmsRun analysis_${RUNNUMBER}_cfg.py > analysis_${RUNNUMBER}_${DATESUFF}.cout
echo "done."
mv debug.log   analysis_${RUNNUMBER}_${DATESUFF}.debug.log
mv info.log    analysis_${RUNNUMBER}_${DATESUFF}.info.log
mv warning.log analysis_${RUNNUMBER}_${DATESUFF}.warning.log
mv error.log   analysis_${RUNNUMBER}_${DATESUFF}.error.log
echo "  Please check the output  : ${SCRATCH}"/analysis_${RUNNUMBER}_${DATESUFF}.*

# mv client file to the output directory
if [ $USECLIENTFILE = False ] && [ $SAVECLIENTFILE = True ] ; then
  CLIENTFILE=$(ls /tmp | grep $RUNNUMBER | grep Client)
  if [ -n "$CLIENTFILE" ] ; then
    mv /tmp/$CLIENTFILE $SCRATCH
  else
    echo "No client file found to copy!"
  fi
fi

# copy raw data to the output directory
RAWFILES=$(ls $DATALOC/closed | grep `printf %08u $RUNNUMBER`)
for file in $RAWFILES; do
  mv $DATALOC/closed/$file $SCRATCH
done

# create a cfg so users can run from raw again (needs the converted root files as input)
cat $BASEDIR/scripts/sourcefromraw_template_cfg.py \
  | sed 's,RUNNUMBER,'$RUNNUMBER',g' \
  | sed 's,DBPART,'$DBPARTITIONNAME',g' \
  > $SCRATCH/sourcefromraw_${RUNNUMBER}_cfg.py
for file in `ls USC*storageManager*dat` ; do
  echo "process.source.fileNames.extend(cms.untracked.vstring('file:`basename $file .dat`.root'))" >> sourcefromraw_${RUNNUMBER}_cfg.py
done

# run conversion, only automatic in TAC
if [ -n "`uname -n | grep cmstracker029`" ] ; then
  /opt/cmssw/scripts/run_conversion.sh $RUNNUMBER
fi
