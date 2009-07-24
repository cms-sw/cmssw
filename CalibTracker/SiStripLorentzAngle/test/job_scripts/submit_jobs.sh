#!/bin/bash

LIST=list_CRAFT_V3P_woBr.txt
#LIST=list_test.txt
NEVENT=-1;
WORK_DIR=$PWD;
MY_TMP="/tmp/sfrosali";
MY_CASTOR_DIR="/castor/cern.ch/user/s/sfrosali/CRAFT_RUN_CHIOCHIA/IDEAL_GEOM/";
TEMPLATE="template_IdealGeom.py"

cd $WORK_DIR;
eval `scramv1 runtime -sh`

NUMBER=1;

for i in `gawk '{print $1}' $LIST`; do

echo $NUMBER;

if  [ $NUMBER -eq 1 ] ;  then
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
fi

FILENAME=`grep $i $LIST | gawk '{print $1}'`;
echo $FILENAME;
echo $FILETAG;

PY="LA_ProfileBooker_"$FILETAG"_"$NUMBER.py;

JOB="Job_LA_"$FILETAG"_"$NUMBER.sh;

cat $TEMPLATE | sed -e "s#FILETAG#$FILETAG#" | sed -e "s#MY_TMP#$MY_TMP#" | sed -e "s#NEVENT#$NEVENT#" | sed -e "s#FILENAME#$FILENAME#" | sed -e "s#NUMBER#$NUMBER#" > $PY

cat job.sh  | sed -e "s#WORK_DIR#$WORK_DIR#" | sed -e "s#MY_CASTOR_DIR#$MY_CASTOR_DIR#" | sed -e "s#MY_TMP#$MY_TMP#" | sed -e"s#FILETAG#$FILETAG#" | sed -e "s#PY#$PY#" | sed -e "s#NUMBER#$NUMBER#" | sed -e "s#JOB#$JOB#" | sed -e"s#DIR_HISTOS#$DIR_HISTOS#" | sed -e"s#DIR_HISTOHARV#$DIR_HISTOHARV#" | sed -e "s#DIR_TREES#$DIR_TREES#" | sed -e"s#DIR_DEBUG#$DIR_DEBUG#" | sed -e "s#DIR_PY#$DIR_PY#" > $JOB

chmod 755 $JOB;

#if [ $NUMBER != 1 ]; then
bsub -q cmscaf $JOB -J LA_CRAFT_test_22X_$NUMBER;
#fi

if [ $# != 0 ] && [ $NUMBER == $1 ];  then
exit
fi

NUMBER=`expr $NUMBER + 1`;

done;
