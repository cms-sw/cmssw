#!/bin/bash

LIST=list_S156.txt
#LIST=test.txt
NUMBER=1;
NEVENT=-1;
WORK_DIR=$PWD;
MY_TMP="/tmp/sfrosali";
MY_CASTOR_DIR="/castor/cern.ch/user/s/sfrosali/test_new_binning/";

cd $WORK_DIR;
eval `scramv1 runtime -sh`

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
DIR_TREES=$DIR"/trees/";
rfmkdir $MY_CASTOR_DIR/$DIR_TREES;
echo $DIR_TREES;
DIR_FITS=$DIR"/fits/";
rfmkdir $MY_CASTOR_DIR/$DIR_FITS;
echo $DIR_FITS;
DIR_DEBUG=$DIR"/debug/";
rfmkdir $MY_CASTOR_DIR/$DIR_DEBUG;
echo $DIR_DEBUG;
DIR_CFG=$DIR"/cfg/";
rfmkdir $MY_CASTOR_DIR/$DIR_CFG;
echo $DIR_CFG;
fi

FILENAME=`grep $i $LIST | gawk '{print $1}'`;
echo $FILENAME;
echo $FILETAG;

CFG="LA_ProfileBooker_"$FILETAG"_"$NUMBER.cfg;

JOB="Job_LA_"$FILETAG"_"$NUMBER.sh;

cat template.cfg | sed -e "s#FILETAG#$FILETAG#" | sed -e "s#MY_TMP#$MY_TMP#" | sed -e "s#NEVENT#$NEVENT#" | sed -e "s#FILENAME#$FILENAME#" | sed -e "s#NUMBER#$NUMBER#" > $CFG

cat job.sh  | sed -e "s#WORK_DIR#$WORK_DIR#" | sed -e "s#MY_CASTOR_DIR#$MY_CASTOR_DIR#" | sed -e "s#MY_TMP#$MY_TMP#" | sed -e "s#FILETAG#$FILETAG#" | sed -e "s#CFG#$CFG#" | sed -e "s#NUMBER#$NUMBER#" | sed -e "s#JOB#$JOB#" | sed -e "s#DIR_HISTOS#$DIR_HISTOS#" | sed -e "s#DIR_TREES#$DIR_TREES#" | sed -e "s#DIR_FITS#$DIR_FITS#" | sed -e "s#DIR_DEBUG#$DIR_DEBUG#" | sed -e "s#DIR_CFG#$DIR_CFG#" > $JOB

chmod 755 $JOB;

#if [ $NUMBER != 1 ]; then
bsub -q cmscaf $JOB -J LA_CSA08_$NUMBER;
#fi

if [ $# != 0 ] && [ $NUMBER == $1 ];  then
exit
fi

NUMBER=`expr $NUMBER + 1`;

done;
