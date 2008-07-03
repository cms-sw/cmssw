#!/bin/bash
# $Id:$

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
  for file in $( ssh $host "ls -1 $CHECK_PATH_OPEN" )
    do
    echo $file;
    if [ `echo $file | grep -v -E '/|.ind$'` ]
        then
        $CHECK_INJECTION_CMD$file;
    fi
    echo;
  done
	
  echo;echo;echo CLOSED FILES
  for file in $( ssh $host "ls -1 $CHECK_PATH_CLOSED" )
    do
    echo $file;
    if [ `echo $file | grep -v -E '/|.ind$'` ]
        then
        $CHECK_INJECTION_CMD$file;
    fi
    echo;
  done
  echo;echo;echo;
done
exit 0
