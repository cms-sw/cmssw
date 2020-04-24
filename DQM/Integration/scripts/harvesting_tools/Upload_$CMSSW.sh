#!/bin/zsh

LOCK=upload.lock
if [ -e $LOCK ]; then
 echo An update is running with pid $(cat $LOCK)
 echo Remove the lock file $LOCK if the job crashed
 exit
else
 echo $$ > $LOCK
fi

types=("data" "MC" "preprod")

## Enter data and MC storage pathes
for ((index=1; index<=${#types[@]}; ++index))
do
if [ $types[index] = "data" ];
then
Castordir=/castor/cern.ch/cms/store/temp/dqm/offline/harvesting_output/data/dqmoffline
fi
if [ $types[index] = "MC" ];
then
Castordir=/castor/cern.ch/cms/store/temp/dqm/offline/harvesting_output/mc/mc
fi
if [ $types[index] = "preprod" ];
then
Castordir=/castor/cern.ch/cms/store/temp/dqm/offline/harvesting_output/data/mc
fi
echo "-------------------------------------"
echo "Start upload of $types[index] files"
echo "-------------------------------------"


for CMSSW in `nsls $Castordir`;
do

CMSSWdir=$Castordir/$CMSSW
## If a CMSSW version is passed, enter only the according subdirectory
if [ $1 ];
then
PassedDir=$Castordir/$1

if [ $CMSSWdir != $PassedDir ];
then
echo Skip $CMSSWdir 
continue
fi
echo Enter $CMSSWdir
fi

for dataset in `nsls $CMSSWdir`
do
datasetdir=$CMSSWdir/$dataset
echo Enter $datasetdir
for run in `nsls $datasetdir`;
do
rundir=$datasetdir/$run
for nevents in `nsls $rundir`;
do
neventsdir=$rundir/$nevents
for section in `nsls $neventsdir`;
do
sectiondir=$neventsdir/$section
for file in `nsls $sectiondir`;
do
rootfile=$sectiondir/$file
size=`rfstat $rootfile | grep Size | perl -pe 's/Size \(bytes\)    \: //'`
if [ $size -ne 0 ];
then
## Definition of ffile changed due to upgrade to crab 2_7_5
##ffile=DQM_V0$(echo $rootfile | perl -pe 's/.*\/DQM_V0// ; s/_1.root/.root/ ; s/_2.root/.root/; s/_3.root/.root/ ; s/_4.root/.root/ ; s/_5.root/.root/ ; s/_1.root/.root/')
ffile=DQM_V0$(echo $rootfile | perl -pe 's/.*\/DQM_V0// ; s/__DQM.*/__DQM.root/')
file_test=`grep -c "$ffile" upload_bookkeeping.txt`
if [ $file_test -eq 0 ];
then
rfcp $rootfile ./$ffile
#echo "Platzhalter fuer ./rfcp ..."
if [ `echo $?` != 0 ];
then
break
else
../VisMonitoring/DQMServer/scripts/visDQMUpload https://cmsweb.cern.ch/dqm/offline $ffile
#echo "Platzhalter fuer ./VisMonitoring ..."
if [ `echo $?` != 0 ];
then
echo "------------------------------------------------------------------------------------------------"
echo $ffile could not be uploaded
echo "------------------------------------------------------------------------------------------------"
rm $ffile
#echo "Platzhalter fuer rm ..."
break
else
echo "------------------------------------------------------------------------------------------------"
echo $ffile is uploaded
echo "------------------------------------------------------------------------------------------------"
echo $ffile >> upload_bookkeeping.txt
#echo "Platzhalter fuer bookkeeping"
dataset_test=`grep -c "$dataset" dataset_bookkeeping.txt`
if [ $dataset_test -eq 0 ];
then
echo $dataset >> dataset_bookkeeping.txt
#echo "Platzhalter fuer bookkeeping"
echo $dataset
fi
fi
rm $ffile
#echo "Platzhalter fuer rm ..."
fi
fi
fi
done
done
done
done
done
done

done

#rfcp upload_bookkeeping.txt /castor/cern.ch/cms/store/caf/user/npietsch/upload_bookkeeping_backup.txt
cp upload_bookkeeping.txt /afs/cern.ch/user/n/npietsch/harvesting_backup/upload_bookkeeping_backup.txt

#rfcp dataset_bookkeeping.txt /castor/cern.ch/cms/store/caf/user/npietsch/dataset_bookkeeping_backup.txt
cp dataset_bookkeeping.txt /afs/cern.ch/user/n/npietsch/harvesting_backup/dataset_bookkeeping_backup.txt

rm -f $LOCK
