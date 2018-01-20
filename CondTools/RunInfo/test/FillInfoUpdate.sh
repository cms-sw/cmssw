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
# Setup CMSSW area and log files
#-------------------------------------
RELEASE=CMSSW_9_2_6
RELEASE_DIR=/afs/cern.ch/work/a/anoolkar/private/
DIR=/afs/cern.ch/work/a/anoolkar/private/CMSSW_9_2_6/src/CondTools/RunInfo/log
LOGFILE=${DIR}/FillInfoTriggerO2O.log
DATEFILE=${DIR}/FillInfoTriggerO2ODate.log
DATE=`date --utc`
OUTFILE="/afs/cern.ch/work/a/anoolkar/private/CMSSW_9_2_6/src/CondTools/RunInfo/log/o2oUpdate_$$.txt"
D=`date +"%m-%d-%Y-%T" --utc`
OUTFILE="/afs/cern.ch/work/a/anoolkar/private/CMSSW_9_2_6/src/CondTools/RunInfo/log/fill_"$D".log"
pushd $RELEASE_DIR/$RELEASE/src/
#@R#export SCRAM_ARCH=slc6_amd64_gcc493
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

#######     ------  popcon  beginning   --------  #######################

echo " " | tee -a $LOGFILE
echo "--------: FillInfo O2O was triggered at :-------- " | tee -a $LOGFILE
echo "$DATE" | tee -a $LOGFILE

#######     ----     getting the previous cron date ############### 
#######     parsing the last line from PopCon DATE log file###### 
LOGDATE=`cat $DATEFILE | awk 'NR ==1 {print $0}'`
TMSLOGDATE=`date --utc -d "$LOGDATE" +%s`
echo "timestamp for the log (last log)" $TMSLOGDATE "corresponding to date" | tee -a $LOGFILE
echo $LOGDATE | tee -a $LOGFILE
rm -f $DATEFILE
echo $DATE > $DATEFILE

pushd $DIR

echo  "We are in: $PWD" | tee -a $LOGFILE

echo "*** Checking the CMSSW environment for the job ***" | tee -a $LOGFILE
set | tee -a $LOGFILE

#- sdg: These cfg were in $RELEASE_DIR/$RELEASE/src/CondTools/Ecal/python
#       but we keep them in this area in order to avoid issues with the release.
t1=$(awk 'NR == 35 {print $4}' /afs/cern.ch/work/a/anoolkar/private/CMSSW_9_2_6/src/CondTools/RunInfo/test/FillInfoPopConAnalyzer.py)
t2=$(expr "$t1" + 3)
sed -i '35s/'"$t1"'/'"$t2"'/' /afs/cern.ch/work/a/anoolkar/private/CMSSW_9_2_6/src/CondTools/RunInfo/test/FillInfoPopConAnalyzer.py
t1=$(awk 'NR == 36 {print $4}' /afs/cern.ch/work/a/anoolkar/private/CMSSW_9_2_6/src/CondTools/RunInfo/test/FillInfoPopConAnalyzer.py)
t2=$(expr "$t1" + 3)
sed -i '36s/'"$t1"'/'"$t2"'/' /afs/cern.ch/work/a/anoolkar/private/CMSSW_9_2_6/src/CondTools/RunInfo/test/FillInfoPopConAnalyzer.py

submit cmsRun /afs/cern.ch/work/a/anoolkar/private/CMSSW_9_2_6/src/CondTools/RunInfo/test/FillInfoPopConAnalyzer.py       

# END OF CHANGES
log "-----------------------------------------------------------------------"
if [ -n "$KILLSWITCH" ]; then
    log "Killswitch activated"
ADDR="http://$HOSTNAME:$JCPORT/urn:xdaq-application:service=jobcontrol/ProcKill?kill=$$"

KILLCMD="curl $ADDR"

log $KILLCMD
$KILLCMD > /dev/null

fi
log DONE
exit 0
