#!/bin/bash
#######     ------    beginning   --------  #######################
RELEASE=CMSSW_3_5_4_DCS
LOGFILE=/nfshome0/popcondev/SiStripJob/SiStripDCS.log
DATEFILE=/nfshome0/popcondev/SiStripJob/SiStripDateDCS.log
DATE=`date --utc`
echo " " | tee -a $LOGFILE
echo "----- new cronjob started for SiStrip  at -----" | tee -a $LOGFILE
echo "$DATE" | tee -a $LOGFILE

#######     ----     getting the previous cron date ############### 
#######     parsing the last line from PopCon DATE log file###### 
LOGDATE=`cat $DATEFILE | awk 'NR ==1 {print $0}'`
TMSLOGDATE=`date --utc -d "$LOGDATE" +%s`
echo "timestamp for the log (last log)" $TMSLOGDATE "corresponding to date" | tee -a $LOGFILE
echo $LOGDATE | tee -a $LOGFILE
rm -f $DATEFILE
echo $DATE > $DATEFILE

#setting up SCRAM environment
source /nfshome0/cmssw2/scripts/setup.sh
cd /nfshome0/popcondev/SiStripJob/${RELEASE}/src/
eval `scramv1 runtime -sh`
#showtags
#echo ${TNS_ADMIN}

YEAR=`date -u +%Y`
MONTH=`date -u +%m`
# Take the blank padded version of the day, otherwise it will start with a 0
# for days before the 10th and it will crash the o2o.
# DAY=`date -u +%d`
DAY=`date -u +%e`
# Remove leading whitespace if any
DAY=`echo $DAY | tr -s " "`
HOUR=`date -u +%H`

RunDir=/nfshome0/popcondev/SiStripJob/${RELEASE}/src/CalibTracker/SiStripDCS/test/
LogDir=${RunDir}/log
 
echo $DAY.$MONTH.$YEAR $HOUR

cd ${RunDir}

# cat ${RunDir}/dcs_o2o_template_cfg.py | sed -e "s@YEAR@$YEAR@g" -e "s@MONTH@$MONTH@g" -e "s@DAY@$DAY@g" -e "s@HOUR@$HOUR@g" > ${RunDir}/dcs_o2o_$DAY$MONTH$YEAR\_$HOUR\_cfg.py
cat ${RunDir}/dcs_o2o_template_cfg.py | sed -e "s@YEAR@$YEAR@g" -e "s@MONTH@$MONTH@g" -e "s@DAY@$DAY@g" -e "s@HOUR@$HOUR@g" > ${RunDir}/dcs_o2o_cfg.py

python ${RunDir}/ManualO2OForAutomaticProcessing.py | tee -a $LOGFILE

#cmsRun ${RunDir}/dcs_o2o_$DAY$MONTH$YEAR\_$HOUR\_cfg.py | tee -a $LOGFILE #${LogDir}/dcs_o2o_$DAY$MONTH$YEAR\_$HOUR.log 
#mv info.log ${LogDir}/info/info_$DAY$MONTH$YEAR\_$HOUR.log
#mv error.log ${LogDir}/error/error_$DAY$MONTH$YEAR\_$HOUR.log
#mv warning.log ${LogDir}/warning/warning_$DAY$MONTH$YEAR\_$HOUR.log
#mv debug.log ${LogDir}/debug/debug_$DAY$MONTH$YEAR\_$HOUR.log
#rm dcs_o2o_$DAY$MONTH$YEAR\_$HOUR\_cfg.py
