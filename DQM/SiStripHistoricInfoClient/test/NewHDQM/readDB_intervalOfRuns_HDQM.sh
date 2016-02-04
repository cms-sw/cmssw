#!/bin/bash 
#
#
# This script extracts from the database the summary informations of the runs which are in the
# interval "firstRun" "lastRun" and creates trend charts.
#
# ./readDB_intervalOfRuns_HDQM.sh xx firstRun lastRun
# (xx to be chosen by the user to designate for instance a week number)
#
# Please modify $baseDir according to your working area
#
# The database parameters and summary informations to extract
# are specified in $baseDir/DQM/SiStripHistoricInfoClient/test/TrendsWithIOVIterator/template_HDQMInspectorSelection_intervalOfRuns.cc
#
# The trend charts are stored in $baseDir/week_$1_CRAFT
#


baseDir=/home/cmstacuser/historicDQM/CMSSW_Releases/CMSSW_3_1_X_2009-04-07-0600/src
lockFile=$baseDir/lockFile


#if [ ! $1 || ! $2 || ! $3] ;
#    then echo "please provide the number of the week you are considering, the first run number, the last run number"
#         echo "./readDB_intervalOfRuns_HDQM.sh 42 65941 66746 "
#         echo "./readDB_intervalOfRuns_HDQM.sh 43 66748 67647 "
#    exit 1
#fi	



cd $baseDir
eval `scramv1 runtime -sh`



[ -e $lockFile ] && echo -e "lockFile " $lockFile "already exists. Process probably already running. If not remove the lockfile." && exit
touch $lockFile
echo -e "=============================================================="
echo " Creating lockFile :"
ls $lockFile
trap "rm -f $lockFile" exit



echo "=========================================================================="
echo " Extract the infos from the DB & do the trend plots for the last $1 runs  "  
echo "=========================================================================="

cat $baseDir/DQM/SiStripHistoricInfoClient/test/TrendsWithIOVIterator/template_HDQMInspectorSelection_intervalOfRuns.cc | sed -e "s@firstRun@$2@g" | sed -e "s@lastRun@$3@g" > $baseDir/testHDQMInspectorSelection_intervalOfRuns.cc
cp $baseDir/DQM/SiStripHistoricInfoClient/test/TrendsWithIOVIterator/rootlogon.C $baseDir/.
root -l -b -q $baseDir/rootlogon.C
root -l -b -q $baseDir/testHDQMInspectorSelection_intervalOfRuns.cc

if [ `ls historicDQM.root` ]; then
root -l -b -q DQM/SiStripHistoricInfoClient/test/TrendsWithIOVIterator/testHDQMGraphAnalysis.cc
fi

cp $baseDir/DQM/SiStripHistoricInfoClient/test/diow.pl .
./diow.pl

rm -rf week_$1_CRAFT

mkdir week_$1_CRAFT
mkdir week_$1_CRAFT/details

mv *gif week_$1_CRAFT/details/
mv week_$1_CRAFT/details/*superimposed* week_$1_CRAFT/
mv week_$1_CRAFT/details/number_of*gif week_$1_CRAFT/
mv week_$1_CRAFT/details/*integrated*gif week_$1_CRAFT/
mv *.C week_$1_CRAFT/details/.
mv historicDQM.root week_$1_CRAFT/details/.

cp $baseDir/DQM/SiStripHistoricInfoClient/test/NewHDQM/index.html week_$1_CRAFT/.


rm -f testHDQMInspectorSelection_intervalOfRuns.cc
rm -f $lockFile
