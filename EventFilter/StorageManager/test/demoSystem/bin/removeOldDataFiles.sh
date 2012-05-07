#!/bin/bash
#
# 24-Jan-2008, KAB - simple script to clean up 
# data files in the SM development system.

cutoffMinutes=10

date
echo "Removing data files older than ${cutoffMinutes} minutes..."
cd $STMGR_DIR/db

find mbox open closed -type f -mmin +${cutoffMinutes} -exec rm -f '{}' \; &
