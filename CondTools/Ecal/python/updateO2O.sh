#!/bin/bash
RELEASE=CMSSW_8_0_1
RELEASE_DIR=/data/O2O/Ecal/cmssw
DIR=/data/O2O/Ecal/TPG
LOGFILE=${DIR}/EcalTriggerO2O.log
DATEFILE=${DIR}/EcalTriggerO2ODate.log
DATE=`date --utc`
OUTFILE="/tmp/o2oUpdate_$$.txt"

echo "*** Checking the environment for the job ***" | tee -a $LOGFILE
set | tee -a $LOGFILE
#setting up environment variables
#export HOME=/nfshome0/popconpro
export HOME=/nfshome0/ecaldb
export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:/nfshome0/popconpro/bin
echo "*** Checking the environment for the job ***" | tee -a $LOGFILE
set | tee -a $LOGFILE

function log() {
    echo "[`date`] : $@ " | tee -a $OUTFILE
}

function submit() {
    log $@
     $@ | tee -a -a $OUTFILE
}

JCPORT=9999

while getopts ":t:r:p:k" options; do
    case $options in
        t ) TPG_KEY=$OPTARG;;
        r ) RUN_NUMBER=$OPTARG;;
        p ) JCPORT=$OPTARG;;
        k ) KILLSWITCH=1;;
    esac
done

log "-----------------------------------------------------------------------"
log "${DIR}/updateO2O.sh"
log "PID $$"
log "HOSTNAME $HOSTNAME"
log "JCPORT $JCPORT"
log "TPG_KEY $TPG_KEY"
log "RUN_NUMBER $RUN_NUMBER"
log "date `date`"
log "-----------------------------------------------------------------------"

# CHANGE HERE
#sleep 2
#echo `date` > $OUTFILE
#echo PID $$ >> $OUTFILE
#echo USER `whoami` >> $OUTFILE
#echo HOSTNAME $HOSTNAME >> $OUTFILE
#echo TPG_KEY $TPG_KEY >> $OUTFILE
#echo RUN_NUMBER $RUN_NUMBER >> $OUTFILE

#######     ------  popcon  beginning   --------  #######################

echo " " | tee -a $LOGFILE
echo "----- new cronjob started for Ecal Trigger O2O at -----" | tee -a $LOGFILE
echo "$DATE" | tee -a $LOGFILE

#######     ----     getting the previous cron date ############### 
#######     parsing the last line from PopCon DATE log file###### 
LOGDATE=`cat $DATEFILE | awk 'NR ==1 {print $0}'`
TMSLOGDATE=`date --utc -d "$LOGDATE" +%s`
echo "timestamp for the log (last log)" $TMSLOGDATE "corresponding to date" | tee -a $LOGFILE
echo $LOGDATE | tee -a $LOGFILE
rm -f $DATEFILE
echo $DATE > $DATEFILE


pushd $RELEASE_DIR/$RELEASE/src/

export SCRAM_ARCH=slc6_amd64_gcc493
source /opt/offline/cmsset_default.sh
eval `scramv1 runtime -sh` 

pushd $DIR

echo  "We are in: $PWD" | tee -a $LOGFILE

echo "*** Checking the CMSSW environment for the job ***" | tee -a $LOGFILE
set | tee -a $LOGFILE

#- sdg: These cfg were in $RELEASE_DIR/$RELEASE/src/CondTools/Ecal/python
#       but we keep them in this area in order to avoid issues with the release.
submit cmsRun copyBadTT_cfg.py       
submit cmsRun copyBadXT_cfg.py       
submit cmsRun copyFgrGroup_cfg.py    
submit cmsRun copyFgrIdMap_cfg.py    
submit cmsRun copyFgrStripEE_cfg.py  
submit cmsRun copyFgrTowerEE_cfg.py  
submit cmsRun copyLin_cfg.py         
submit cmsRun copyLutGroup_cfg.py    
submit cmsRun copyLutIdMap_cfg.py    
submit cmsRun copyPed_cfg.py         
submit cmsRun copyPhysConst_cfg.py   
submit cmsRun copySli_cfg.py         
submit cmsRun copyWGroup_cfg.py      
submit cmsRun copyWIdMap_cfg.py      
submit cmsRun copySpikeTh_cfg.py
submit cmsRun copyBadStrip_cfg.py
submit cmsRun updateIntercali_express.py
submit cmsRun updateIntercali_hlt.py
submit cmsRun updateADCToGeV_express.py
submit cmsRun updateADCToGeV_hlt.py


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
