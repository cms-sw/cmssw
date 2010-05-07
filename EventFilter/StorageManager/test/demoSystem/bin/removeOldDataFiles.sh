#!/bin/bash
#
# 24-Jan-2008, KAB - simple script to clean up 
# data files in the SM development system.

cutoffDays=1

date
echo "Removing data files older than ${cutoffDays} days..."
cd $STMGR_DIR/db

find mbox open closed -type f -mtime +${cutoffDays} -print -exec rm -f '{}' \;
