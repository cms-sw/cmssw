#!/bin/bash

DirRootFiles='/eos/cms/store/user/zhokin/PSM/HcalNZS/2018'

echo $DirRootFiles
fileName=$1
echo $fileName
DIROUT=$2
echo $DIROUT

# Obtain the runList from a file, if needed
#runList=""
runList=`cat ${fileName}`
for i in ${runList} ; do
NRUN=$i
echo $NRUN

list=`ls $DirRootFiles/$NRUN/HcalNZS/crab_*_*/*_*/0000/Global_*.root`
echo $list
echo " start hadd   "
hadd Global_$NRUN.root $list


echo " start copying"
mv Global_$NRUN.root $DIROUT/.

done

