#!/bin/bash
#
# 24-Jan-2008, KAB - simple script to clean up 
# log files in the SM development system.

cutoffDays=1

date
cd $STMGR_DIR/log

echo "Removing log files older than ${cutoffDays} days..."
find . -type f -mtime +${cutoffDays} -name "*log" -print -exec rm -f '{}' \;

echo "Removing core files..."
find . -type f -name "core.*" -print -exec rm -f '{}' \;
