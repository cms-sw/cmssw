#!/bin/bash
#
# 24-Jan-2008, KAB - simple script to clean up 
# data files in the SM development system.

cutoffDays=14

date
echo "Removing data files older than ${cutoffDays} days..."
cd $STMGR_DIR/db

# find the old files in the directories that we want to clean
oldFileList=`find mbox open closed -type f -mtime +${cutoffDays} -print`

# remove the files
for oldFile in $oldFileList
do
    rm -fv "$oldFile"
done
