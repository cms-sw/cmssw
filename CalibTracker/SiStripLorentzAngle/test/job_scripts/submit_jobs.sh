#!/bin/bash

#LIST=list_CRAFT_repro_TOTAL.txt
LIST=source_list_test.txt
NEVENT=-1;
WORK_DIR=$PWD;
MY_TMP="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/LA_calibration/CMSSW_3_1_0_pre7/src/CalibTracker/SiStripLorentzAngle/test/Test_LAProfileBooker/tmp";
MY_CASTOR_DIR="/castor/cern.ch/user/s/sfrosali/CRAFT_REPRO/CRAFT_REPRO_NEWAL/TEST_31X";
TEMPLATE="template_100J_NEWAL.py"
JOB_REPORT=0;

cd $WORK_DIR;
eval `scramv1 runtime -sh`

NN=1;
NUMBER=1;
for i in `gawk '{print $1}' $LIST`; do

#echo $NN;
echo $i >> LIST_CRAFT_100J_$NUMBER.txt;
NN=`expr $NN + 1`;

if  [ $NN -eq 100 ] ;  then
echo "Compilata lista numero = " $NUMBER
NUMBER=`expr $NUMBER + 1`;
NN=1;
fi

done;

echo "Compilata lista numero = " $NUMBER

mkdir CRAFT_Repro_Lists;

DIR=`grep $i $LIST | gawk -F / '{print $4}'`;
rfmkdir $MY_CASTOR_DIR/$DIR;
DIR=$DIR"/"`grep $i $LIST | gawk -F / '{print $5}'`;
rfmkdir $MY_CASTOR_DIR/$DIR;
DIR=$DIR"/"`grep $i $LIST | gawk -F / '{print $6}'`;
rfmkdir $MY_CASTOR_DIR/$DIR;
DIR=$DIR"/"`grep $i $LIST | gawk -F / '{print $7}'`;
rfmkdir $MY_CASTOR_DIR/$DIR;
FILETAG=`grep $i $LIST | gawk -F / '{print $7}'`;
DIR_HISTOS=$DIR"/histos/";
rfmkdir $MY_CASTOR_DIR/$DIR_HISTOS;
echo $DIR_HISTOS;
DIR_HISTOHARV=$DIR"/histos_Harv/";
rfmkdir $MY_CASTOR_DIR/$DIR_HISTOHARV;
echo $DIR_HISTOHARV;
DIR_TREES=$DIR"/trees/";
rfmkdir $MY_CASTOR_DIR/$DIR_TREES;
echo $DIR_TREES;
DIR_DEBUG=$DIR"/debug/";
rfmkdir $MY_CASTOR_DIR/$DIR_DEBUG;
echo $DIR_DEBUG;
DIR_PY=$DIR"/py/";
rfmkdir $MY_CASTOR_DIR/$DIR_PY;
echo $DIR_PY;

for (( i = 1; i <= $NUMBER; i++ )); do
echo "Sottomissione job n = " $i;

#echo $FILETAG;

PY="LA_PB_100J_"$i.py;

JOB="Job_LA_100J_"$i.sh;

JLIST=LIST_CRAFT_100J_$i.txt;

cat $TEMPLATE | sed -e "s#JLIST#$JLIST#" | sed -e "s#MY_TMP#$MY_TMP#" | sed -e "s#NEVENT#$NEVENT#"  | sed -e "s#NUMBER#$i#" > $PY

cat job_100J_NEWAL.sh   | sed -e "s#JLIST#$JLIST#" | sed -e "s#WORK_DIR#$WORK_DIR#" | sed -e "s#MY_CASTOR_DIR#$MY_CASTOR_DIR#" | sed -e "s#MY_TMP#$MY_TMP#" | sed -e "s#PY#$PY#" | sed -e "s#NUMBER#$i#" | sed -e "s#JOB#$JOB#" | sed -e"s#DIR_HISTOS#$DIR_HISTOS#" | sed -e"s#DIR_HISTOHARV#$DIR_HISTOHARV#" | sed -e "s#DIR_TREES#$DIR_TREES#" | sed -e"s#DIR_DEBUG#$DIR_DEBUG#" | sed -e "s#DIR_PY#$DIR_PY#" > $JOB

chmod 755 $JOB;

#if [ $JOB_REPORT -eq 0 ]; then
#bsub -o $JOB_REP -q cmscaf $JOB -J LA_CRAFT_test_Harvesting_$i;
#fi
#if [ $JOB_REPORT -eq 1 ]; then
bsub -q cmscaf $JOB -J LA_$i;
#fi

#if [ $# != 0 ] && [ $N == $1 ];  then
#exit
#fi

done;

