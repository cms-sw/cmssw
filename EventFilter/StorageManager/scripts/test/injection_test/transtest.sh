#!/bin/bash
# Modified injbyhand.sh for testing with sym links and hardcoded parameters

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

setuplabel=TransferTest20080625
stream=TransferTest20080625
hname=`hostname | cut -d. -f1` 
size=`stat -L -c "%s" $name`  
ctime=`stat -c "%Y" $name`
itime=`date +%s`

echo Path:       $pname
echo File:       $fname
echo SetupLabel: $setuplabel
echo Stream:     $stream
echo Size:       $size
echo Ctime:      $ctime
echo Itime:      $itime

tier0="/nfshome0/tier0/scripts/switched_off_for_today.pl"
$debug $tier0 --filename $fname --path $pname --filesize $size --type streamer \
    --hostname $hname --destination TransferTest --producer StorageManager --appname TestApp \
    --appversion v1 --runnumber 12345  --lumisection 1 \
    --count 0 --stream $stream  --instance 0 --setuplabel $setuplabel \
    --nevents 1 --ctime $ctime --itime $itime  \
    --checksum 0 --comment 'Injected for transfer tests'
