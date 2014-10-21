#!/bin/zsh

LOCK=upload.lock
if [ -e $LOCK ]; then
 echo An update is running with pid $(cat $LOCK)
 echo Remove the lock file $LOCK if the job crashed
 exit
else
 echo $$ > $LOCK
fi

date=`date`


echo
echo "============================"
echo "running Upload at" $date
echo
echo "certificate:"
source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh
X509_USER_PROXY=$HOME/x509up
#
# note: before installing cronjob, run voms-proxy-init -hours=100000 with X509_USER_PROXY set to sth not on /tmp
#
voms-proxy-info

echo 
echo "----------------------------"
for i in "data/dqmoffline" "mc/mc" 
do
echo 
echo "==== Upload of new files in" $i
Castordir=/castor/cern.ch/cms/store/temp/dqm/offline/harvesting_output/$i
for CMSSW in `nsls $Castordir`;
do
CMSSWdir=$Castordir/$CMSSW
for dataset in `nsls $CMSSWdir`
do
datasetdir=$CMSSWdir/$dataset
echo $CMSSW ":" $dataset
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
rfcp $rootfile /tmp/$ffile
if [ `echo $?` != 0 ];
then
break
else
../VisMonitoring/DQMServer/scripts/visDQMUpload https://cmsweb.cern.ch/dqm/offline /tmp/$ffile
if [ `echo $?` != 0 ];
then
echo "------------------------------------------------------------------------------------------------"
echo /tmp/$ffile could not be uploaded
echo "------------------------------------------------------------------------------------------------"
rm -f /tmp/$ffile
break
else
echo "------------------------------------------------------------------------------------------------"
echo $ffile is uploaded
echo "------------------------------------------------------------------------------------------------"
echo $ffile >> upload_bookkeeping.txt


dataset_test=`grep -c "$dataset" dataset_bookkeeping.txt`
if [ $dataset_test -eq 0 ];
then
echo $dataset >> dataset_bookkeeping.txt
echo $dataset
fi

fi
rm -f $ffile
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

cp upload_bookkeeping.txt /afs/cern.ch/user/n/npietsch/harvesting_backup/upload_bookkeeping_backup.txt

cp dataset_bookkeeping.txt /afs/cern.ch/user/n/npietsch/harvesting_backup/dataset_bookkeeping_backup.txt

echo "done upload"
echo
echo "============================"
echo "... done Upload of" $date 
echo "                at" `date`
echo

rm -f $LOCK
