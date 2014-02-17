#!/bin/bash 
#
# This script extracts the summary informations of the last xx runs from 
# the database and creates trend charts.
#
# ./readDB_lastRuns_HDQM.sh xx
#
# modify $baseDir according to your working area
#
# The database parameters and summary informations to extract
# are specified in $baseDir/DQM/SiStripHistoricInfoClient/test/TrendsWithIOVIterator/template_HDQMInspectorSelection_lastRuns.cc
#
# The trend charts are stored in $baseDir/CRAFT_last_$1_runs
#


baseDir=/home/cmstacuser/historicDQM/CMSSW_Releases/CMSSW_3_1_X_2009-04-07-0600/src
lockFile=$baseDir/lockFile


if [ ! $1 ] ;
    then echo "please provide the number of runs you would like to consider..."
         echo "./readDB_lastRuns_HDQM.sh 10 "
    exit 1
fi	



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

cat $baseDir/DQM/SiStripHistoricInfoClient/test/TrendsWithIOVIterator/template_HDQMInspectorSelection_lastRuns.cc | sed -e "s@nRuns@$1@g" > $baseDir/testHDQMInspectorSelection_lastRuns.cc
cp  $baseDir/DQM/SiStripHistoricInfoClient/test/TrendsWithIOVIterator/rootlogon.C $baseDir/.
root -l -b -q $baseDir/rootlogon.C
root -l -b -q $baseDir/testHDQMInspectorSelection_lastRuns.cc

if [ `ls historicDQM.root` ]; then
root -l -b -q DQM/SiStripHistoricInfoClient/test/TrendsWithIOVIterator/testHDQMGraphAnalysis.cc
fi

cp $baseDir/DQM/SiStripHistoricInfoClient/test/diow.pl .
./diow.pl

rm -rf CRAFT_last_$1_runs
mkdir CRAFT_last_$1_runs
mkdir CRAFT_last_$1_runs/details


mv *gif CRAFT_last_$1_runs/details/
mv CRAFT_last_$1_runs/details/*superimposed* CRAFT_last_$1_runs/
mv CRAFT_last_$1_runs/details/number_of*gif CRAFT_last_$1_runs/
mv CRAFT_last_$1_runs/details/*integrated*gif CRAFT_last_$1_runs/
mv *.C CRAFT_last_$1_runs/details/.
mv historicDQM.root CRAFT_last_$1_runs/details/.

cp $baseDir/DQM/SiStripHistoricInfoClient/test/NewHDQM/index.html CRAFT_last_$1_runs/.


rm -f testHDQMInspectorSelection_lastRuns.cc
rm -f $lockFile
