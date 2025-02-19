#!/bin/bash
#
# 17-Jun-2008, KAB - simple script to clean up 
# DQM files in the SM development system.

cutoffDays=1

date
echo "Removing DQM files older than ${cutoffDays} days..."
cd $STMGR_DIR

# find the old files in the directories that we want to clean
find smDQM smpsDQM -type f -mtime +${cutoffDays} -print -exec rm -f '{}' \;
