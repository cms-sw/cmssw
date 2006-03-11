#!/bin/bash

if [ $# -lt 3 ]
then
	echo "arguments: function_name_file pathes_file path_id"
	exit 1
fi

FUNCS=$1
PATHS=$2
ID=$3

# echo $#

# while read a
# do

	a=`grep "^$ID " $PATHS`
	if [ $? != 0 ]
	then
		echo "$ID not found"
		exit 1
	fi

	set -- $a
	shift
	shift
	# echo $*
	for num in $*
	do
		# echo $num
		e=`grep "^$num " $FUNCS`
		echo $e
	done
	echo "-------------------------------------------"
# done
