#!/bin/bash
#
# 22-Aug-2007, KAB - simple script to clean up 
# log files in the SM playback system.

# this value needs to be longer than the forced restart time in the watchdog
# script so that we don't accidentally delete the current logfile
cutoffDays=14

date
echo "Removing log files older than ${cutoffDays} days..."
cd $SMPB_DIR/log

# find the old files in the directories that we want to clean
oldFileList=`find consumer builderUnit filterUnit storageManager -type f -mtime +${cutoffDays} -print | grep 'log$'`

# remove the files
for oldFile in $oldFileList
do
    rm -v "$oldFile"
done
