#!/bin/bash
# $Id: check_left_files_opensize.sh,v 1.2 2008/07/03 10:46:58 loizides Exp $

# This script iterates over all files in the list of hosts provided and calls
# injection script with --check option to show a status report.

# Have a look at the configuration file
source check_left_files.cfg

if [ ! -r $HOSTLIST ]
then
    echo "ERROR: File $HOSTLIST missing!"
    exit 1
fi
echo "File $HOSTLIST is 0K"
echo;echo;echo;

for host in $( cat $HOSTLIST )
do
  echo Processing $host:

  echo;echo;echo OPEN FILES
  ssh $host "du -h $CHECK_PATH_OPEN"
#  echo;echo;echo CLOSED FILES
#  ssh $host "du -h $CHECK_PATH_CLOSED"
done
exit 0
