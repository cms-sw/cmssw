#!/bin/bash

export FILE=$1
export DEST=$2

if [ -f $DEST ]; then
    echo ${DEST} ' exists  ... OK'
else    
    echo 'Copying ' ${DEST} 
    cp $FILE $DEST
    if [ -f ${DEST} ]; then
	echo '  ... OK'  
    fi
fi
