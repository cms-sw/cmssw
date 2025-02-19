baseDir=/home/cmstacuser/historicDQM/CMSSW_Releases/CMSSW_3_1_X_2009-03-31-0100/src
lockFile=$baseDir/lockFile



if [ ! $1 || ! $2 || ! $3] ;
    then echo "please provide the number of the week you are considering, the first run number, the last run number"
         echo "./readDB_intervalOfRuns_HDQM.sh 42 65941 66746 "
       # echo "./readDB_intervalOfRuns_HDQM.sh 43 66748 67647 "
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

cat $baseDir/DQMServices/Diagnostic/test/template_HDQMInspectorSelection_intervalOfRuns.cc | sed -e "s@firstRun@$2@g" | sed -e "s@lastRun@$3@g" > $baseDir/testHDQMInspectorSelection_intervalOfRuns.cc
cp $baseDir/DQMServices/Diagnostic/test/rootlogon.C $baseDir/.
root -l -b -q $baseDir/rootlogon.C
root -l -b -q $baseDir/testHDQMInspectorSelection_intervalOfRuns.cc

if [ `ls historicDQM.root` ]; then
root -l -b -q DQMServices/Diagnostic/test/testGraphAnalysis.cc
fi

#cp $baseDir/DQMServices/Diagnostic/test/diow.pl .
#./diow.pl

rm -rf week_$1_CRAFT

mkdir week_$1_CRAFT
mkdir week_$1_CRAFT/details
mkdir week_$1_CRAFT/trends_by_layer_TIB
mkdir week_$1_CRAFT/trends_by_layer_TOB
mkdir week_$1_CRAFT/trends_by_layer_TID_Side1
mkdir week_$1_CRAFT/trends_by_layer_TID_Side2
mkdir week_$1_CRAFT/trends_by_layer_TEC

mv *TIB* week_$1_CRAFT/trends_by_layer_TIB
mv *TOB* week_$1_CRAFT/trends_by_layer_TOB
mv *TID_Side1* week_$1_CRAFT/trends_by_layer_TID_Side1
mv *TID_Side2* week_$1_CRAFT/trends_by_layer_TID_Side2
mv *TIDLayers* week_$1_CRAFT/trends_by_layer_TID_Side1
mv *TEC* week_$1_CRAFT/trends_by_layer_TEC
mv *gif week_$1_CRAFT/details/
mv week_$1_CRAFT/details/*superimposed* week_$1_CRAFT/
mv week_$1_CRAFT/details/number_of*gif week_$1_CRAFT/
mv week_$1_CRAFT/details/mean_number_of_tracks_per_event*.gif week_$1_CRAFT/
mv week_$1_CRAFT/details/*integrated*gif week_$1_CRAFT/
mv *.C week_$1_CRAFT/details/.
mv historicDQM.root week_$1_CRAFT/details/.

#cp $baseDir/DQMServices/Diagnostic/test/index.html week_$1_CRAFT/.


rm -f testHDQMInspectorSelection_intervalOfRuns.cc
rm -f $lockFile
