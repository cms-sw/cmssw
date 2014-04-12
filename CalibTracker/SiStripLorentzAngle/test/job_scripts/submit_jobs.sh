#!/bin/bash

#List of source files
#LIST=list_CRAFT_repro_TOTAL.txt
LIST=source_list_test.txt

#Number of events to be analyzed
NEVENT=-1;

#Directories used by the jobs
WORK_DIR=$PWD;
MY_TMP=$PWD"/tmp";
MY_CASTOR_DIR="/castor/cern.ch/user/s/sfrosali/CRAFT_REPRO/CRAFT_REPRO_NEWAL/TEST_31X";
mkdir $MY_TMP;

#Name of the queue where the jobs will be submitted
JOB_QUEUE="8nh";

#Name of the templates used to create py and sh files for the jobs
PY_TEMPLATE="template.py";
JOB_TEMPLATE="job.sh";

#Global tag used for reconstruction
GLOBAL_TAG="GR09_31X_V3P";

#Name of the outputs (histos, trees ...)
MY_HISTOS="LA_Histos";
MY_TREE="LA_Tree";
MY_DEBUG="LAProfileDebug";
MY_HISTOS_HARV="LA_Histos_Harv";

cd $WORK_DIR;
eval `scramv1 runtime -sh`

NN=1;
NUMBER=1;
for i in `gawk '{print $1}' $LIST`; do

echo $i >> Source_List_100J_$NUMBER.txt;
NN=`expr $NN + 1`;

if  [ $NN -eq 100 ] ;  then
echo "List number "$NUMBER" compiled"
NUMBER=`expr $NUMBER + 1`;
NN=1;
fi

done;

echo "List number "$NUMBER" compiled"

mkdir Source_Lists;

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
echo "Job n = "$i" submitted";

PY="LA_PB_100J_"$i.py;

JOB="Job_LA_100J_"$i.sh;

JLIST=Source_List_100J_$i.txt;

cat $PY_TEMPLATE     | sed -e "s#GLOBAL_TAG#$GLOBAL_TAG#" | sed -e "s#MY_TREE#$MY_TREE#" | sed -e "s#MY_DEBUG#$MY_DEBUG#" | sed -e "s#MY_HISTOS_HARV#$MY_HISTOS_HARV#" | sed -e "s#MY_HISTOS#$MY_HISTOS#" | sed -e "s#JLIST#$JLIST#" | sed -e "s#MY_TMP#$MY_TMP#" | sed -e "s#NEVENT#$NEVENT#"  | sed -e "s#NUMBER#$i#" > $PY

cat $JOB_TEMPLATE    | sed -e "s#MY_TREE#$MY_TREE#" | sed -e "s#MY_DEBUG#$MY_DEBUG#" | sed -e "s#MY_HISTOS_HARV#$MY_HISTOS_HARV#" | sed -e "s#MY_HISTOS#$MY_HISTOS#" | sed -e "s#JLIST#$JLIST#" | sed -e "s#WORK_DIR#$WORK_DIR#" | sed -e "s#MY_CASTOR_DIR#$MY_CASTOR_DIR#" | sed -e "s#MY_TMP#$MY_TMP#" | sed -e "s#PY#$PY#" | sed -e "s#NUMBER#$i#" | sed -e "s#JOB#$JOB#" | sed -e"s#DIR_HISTOS#$DIR_HISTOS#" | sed -e"s#DIR_HISTOHARV#$DIR_HISTOHARV#" | sed -e "s#DIR_TREES#$DIR_TREES#" | sed -e"s#DIR_DEBUG#$DIR_DEBUG#" | sed -e "s#DIR_PY#$DIR_PY#" > $JOB

chmod 755 $JOB;

bsub -q $JOB_QUEUE $JOB -J LA_$i;

done;

