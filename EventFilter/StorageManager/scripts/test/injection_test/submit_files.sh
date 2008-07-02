#!/bin/bash
# This utility script will submit all files created in /dev/shm
source txtst.cfg

echo "Number iterations: $NUMFILES"

if [ ! -r $HOSTLIST ]
then
	echo "ERROR: File $HOSTLIST missing!"
	exit 1
fi

echo "File $HOSTLIST is 0K"

for host in $( cat $HOSTLIST )
do
	echo Processing $host:
	ssh $host  "
		for i in \$( seq 1 $NUMFILES);
		do
		  if [ ! -h /dev/shm/'$FILEPREFIX'_\$( date +%F)#'$TSTNUM'_'$host'_\$i ];
		    then echo "ERROR: Data file missing!";exit 1;
		  fi;
		  '$SUBMIT_CMD' /dev/shm/'$FILEPREFIX'_\$( date +%F)#'$TSTNUM'_'$host'_\$i;
		  sleep '$SLEEP_TIME'
		done &> '$LOG_PATH'\submit_'$host'_$( date +%F)#'$TSTNUM'.log &
	"
	echo;
done
exit 0
