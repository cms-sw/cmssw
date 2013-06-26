#!/bin/bash
# $Id: injbyhand.sh,v 1.2 2008/07/04 13:31:18 loizides Exp $
#
# Script that extracts necessary information for transfer from written file
# and injects the file in to DB using the standalone script.
#

name=$1
if test -z $name; then 
    echo "Specify valid pathname"
    exit 1; 
fi

debug=echo
if test -n $2; then 
    debug=
fi

if ! test -e $name; then
    echo "$name was not found."
    exit 2;
fi
pname=`dirname $name`
fname=`basename $name`
index=$pname/`basename $name .dat`.ind 
if ! test -e $index; then
    echo "$index was not found."
    exit 3;
fi
ifile=`basename $name .dat`.ind 

setuplabel=`echo $fname | cut -d. -f1`
runno=`echo $fname | cut -d. -f2`
lumi=`echo $fname | cut -d. -f3`
stream=`echo $fname | cut -d. -f4`
inst=`echo $fname | cut -d. -f6`
fc=`echo $fname | cut -d. -f7`
runno=`expr $runno \* 1`
lumi=`expr $lumi \* 1`
inst=`expr $inst \* 1`
fc=`expr $fc \* 1`
hname=`hostname | cut -d. -f1` 
size=`stat -c "%s" $name`  
ctime=`stat -c "%Y" $name`
itime=`date +%s`

echo Path:       $pname
echo File:       $fname
echo Index:      $ifile
echo SetupLabel: $setuplabel
echo Run:        $runno
echo Lumi:       $lumi
echo Stream:     $stream
echo Inst:       $inst
echo FC:         $fc
echo Size:       $size
echo Ctime:      $ctime
echo Itime:      $itime

tier0="/nfshome0/tier0/scripts/injectFileIntoTransferSystem.pl"
result=`$tier0 --check --filename $fname`;
echo 
echo "Query on file resulted in: $result";
echo

if test -n "`echo $result | grep 'File not found in database'`"; then
    $debug $tier0 --filename $fname --path $pname --filesize $size --type streamer \
        --hostname $hname --destination Global --producer StorageManager --appname CMSSW \
        --appversion CMSSW_2_0_8_ONLINE1-cms2 --runnumber $runno  --lumisection $lumi \
        --count $fc --stream $stream  --instance $inst --setuplabel $setuplabel \
        --nevents 1 --ctime $ctime --itime $itime --index $ifile \
        --checksum 0 --comment 'Injected with injbyhand.sh';
elif test -n "`echo $result | grep FILES_INJECTED`"; then
    notscript="/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh"
    $debug $notscript --APP_NAME CMSSW --APP_VERSION CMSSW_2_0_8_ONLINE1-cms2 --RUNNUMBER $runno \
        --LUMISECTION $lumi  --START_TIME $ctime \
        --STOP_TIME $ctime --FILENAME $fname --PATHNAME $pname --HOSTNAME $hname \
        --DESTINATION Global --SETUPLABEL $setuplabel --STREAM $stream --TYPE streamer \
        --NEVENTS 1 --FILESIZE $size --CHECKSUM 0 --INDEX $ifile
elif test -n "`echo $result | grep FILES_TRANS_CHECKED`"; then
    echo "Nothing to be done. File is already transferred and checked.";
    exit 0;
else
    echo "Cowardly refusing to do anything."
    exit 1;
fi
