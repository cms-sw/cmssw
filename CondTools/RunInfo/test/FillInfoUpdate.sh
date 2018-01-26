# Basic setup at lxplus using acron
# Ref: http://information-technology.web.cern.ch/services/fe/afs/howto/authenticate-processes
# (a) kinit
# (b) acrontab -e

# The acrontab acript will look like:
# */1 * * * * lxplus049 $PATH/FillInfoUpdate.sh > $PATH/cron_log.txt
# Ref: https://raw.githubusercontent.com/cms-sw/cmssw/09c3fce6626f70fd04223e7dacebf0b485f73f54/RecoVertex/BeamSpotProducer/scripts/READMEMegascript.txt

#####################################
# SHELL SCRIPT TO BE RUN BY ACRON
# Ref: https://github.com/cms-sw/cmssw/blob/09c3fce6626f70fd04223e7dacebf0b485f73f54/CondTools/Ecal/python/updateO2O.sh
#####################################

#-------------------------------------
# Setup CMSSW area and variables
#-------------------------------------
RELEASE_DIR=$(echo $CMSSw_BASE)
TEST_DIR=$PWD
mkdir -p $TEST_DIR/log
LOG_DIR=${TEST_DIR}/log
LOGFILE=${LOG_DIR}/FillInfoTriggerO2O.log
DATEFILE=${LOG_DIR}/FillInfoTriggerO2ODate.log
DATE=`date --utc`
LOG_DATE=`date +"%Y%m%d_%H%M%S" --utc`

#-------------------------------------
# Fetch fill number from previous run.
#-------------------------------------
interval=3
loc_1=$(grep -n firstFill FillInfoPopConAnalyzer.py | cut -d: -f1)
firstfill=$(awk 'NR == '"$loc_1"' {print $4}' ${TEST_DIR}/FillInfoPopConAnalyzer.py)
loc_2=$(grep -n lastFill FillInfoPopConAnalyzer.py | cut -d: -f1)
lastfill=$(awk 'NR == '"$loc_2"' {print $4}' ${TEST_DIR}/FillInfoPopConAnalyzer.py)
sed -i ''"$loc_1"'s/'"$firstfill"'/'`expr $lastfill + 1`'/' ${TEST_DIR}/FillInfoPopConAnalyzer.py
sed -i ''"$loc_2"'s/'"$lastfill"'/'`expr $lastfill + $interval`'/' ${TEST_DIR}/FillInfoPopConAnalyzer.py
let "firstfill=lastfill+1"
let "lastfill=lastfill+interval"

#-------------------------------------
# Setup CMSSW log files
#-------------------------------------
OUTFILE="${TEST_DIR}/log/fill_"$LOG_DATE"_"$firstfill"-"$lastfill".log"
pushd $RELEASE_DIR/src/
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh` 

#-------------------------------------
# Define various functions
#-------------------------------------
function log() {
    echo "[`date`] : $@ " | tee -a $OUTFILE
}

function submit() {
    log $@
     $@ | tee -a -a $OUTFILE
}

#-------------------------------------
# Get previous triggering date
#-------------------------------------
echo "--------: FillInfo O2O was triggered at :-------- " | tee -a $LOGFILE
echo "$DATE" | tee -a $LOGFILE
LOGDATE=`cat $DATEFILE | awk 'NR ==1 {print $0}'`
TMSLOGDATE=`date --utc -d "$LOGDATE" +%s`
echo "timestamp for the log (last log)" $TMSLOGDATE "corresponding to date" | tee -a $LOGFILE
echo $LOGDATE | tee -a $LOGFILE
rm -f $DATEFILE
echo $DATE > $DATEFILE
pushd $TEST_DIR


#-------------------------------------
# Run FillInfoPopConAnalyzer.py 
#-------------------------------------
submit cmsRun FillInfoPopConAnalyzer.py       
log DONE
exit 0
