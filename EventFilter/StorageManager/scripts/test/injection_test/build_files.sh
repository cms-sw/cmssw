#!/bin/bash
# This utility script creates a file in /dev/shm and as many soft links as requested

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
		if [ ! -r /dev/shm/'$FILEPREFIX'_'$host' ];
		then 
		  echo " creating new data file";
		  dd if=/dev/urandom of=/dev/shm/'$FILEPREFIX'_'$host' bs=64K count=16K &> /dev/null;
		fi;

		echo " cleaning all links";
		find /dev/shm/ -type l | xargs rm &> /dev/null;

		echo " creating new links ";
		for i in \$( seq 1 $NUMFILES);
		do
		  ln -s /dev/shm/'$FILEPREFIX'_'$host' /dev/shm/'$FILEPREFIX'_\$( date +%F )#'$TSTNUM'_'$host'_\$i;
		  echo -n ".";
		done;
	";
	echo;
done
exit 0
