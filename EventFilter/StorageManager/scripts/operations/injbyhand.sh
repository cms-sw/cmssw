#!/bin/bash
# $Id:$
#
# Script that extracts necessary information for transfer from written file
# and injects the file in to DB using the standalone script.
#

doinject=1
#debug=echo

name=$1
if test -z $name; then 
    echo "Specify valid pathname"
    exit 1; 
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

if test "$inject" = "1"; then
    tier0="/nfshome0/tier0/scripts/injectFileIntoTransferSystem.pl"
    $debug $tier0 --filename $fname --path $pname --filesize $size --type streamer \
        --hostname $hname --destination Global --producer StorageManager --appname CMSSW \
        --appversion CMSSW_2_0_8_ONLINE1-cms2 --runnumber $runno  --lumisection $lumi \
        --count $fc --stream $stream  --instance $inst --setuplabel $setuplabel \
        --nevents 1 --ctime $ctime --itime $itime --index $ifile \
        --checksum 0 --comment 'Injected by hand'
else
    notscript="/nfshome0/cmsprod/TransferTest/injection/sendNotification.sh"
    $debug $notscript --APP_NAME CMSSW --APP_VERSION CMSSW_2_0_8_ONLINE1-cms2 --RUNNUMBER $runno \
        --LUMISECTION $lumi  --START_TIME $ctime \
        --STOP_TIME $ctime --FILENAME $fname --PATHNAME $pname --HOSTNAME $hname \
        --DESTINATION Global --SETUPLABEL $setuplabel --STREAM $stream --TYPE streamer \
        --NEVENTS 1 --FILESIZE $size --CHECKSUM 0 --INDEX $ifile
fi
