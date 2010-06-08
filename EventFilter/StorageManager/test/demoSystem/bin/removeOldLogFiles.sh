#!/bin/bash
#
# 24-Jan-2008, KAB - simple script to clean up 
# log files in the SM development system.

cutoffDays=14

date
echo "Removing log files older than ${cutoffDays} days..."
cd $STMGR_DIR/log

# find the old files in the directories that we want to clean
oldFileList=`find builderUnit client client1 client2 consFU filterUnit smProxy storageManager -type f -mtime +${cutoffDays} -print | grep 'log$'`

# remove the files
for oldFile in $oldFileList
do
    rm -v "$oldFile"
done
