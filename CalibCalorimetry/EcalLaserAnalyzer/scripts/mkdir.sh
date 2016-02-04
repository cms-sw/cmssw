#!/bin/bash

export MYDIR=$1
export MESSAGE=$2
echo ${MESSAGE} ': ' 
echo $MYDIR
if [ -d $MYDIR ]; then
    echo '    ... OK'
else
    echo '    ... does not exist -- create'
    mkdir $MYDIR
    if [ -d $MYDIR ]; then
	echo '    ... OK'
    else
	echo '    ... creation failed'
	return
    fi
fi

