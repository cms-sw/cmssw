#!/bin/bash
#
# Make sure the user has given at least one file name
#
if [ $# -lt 1 ]
then
	echo " "
	echo "Please supply a file name as an argument."
	exit
fi
#
# Has the user set up some version of Root?
#
if [ -z "$ROOTSYS" ]
then
	echo " "
	echo "No version of Root is set up. Aborting."
	exit
fi
#
# Cycle through the list of input files one by one. If the
# file exists, process it.  Otherwise complain and move on.
#
while [ $# != 0 ]
do
	if [ -f $1 ]
	then
		echo " "
		echo "Processing file $1"
		echo " "
		root -l << EOF
.x inventory.C+
$1
quit
.q
EOF
	else
		echo " "
		echo "There is no file named $1.  Skipping."
	fi
	shift
done
#
exit
