#!/bin/bash

export MYDIR=$1
export MYLINK=$2
echo 'Creating link '$MYLINK

if [ -d $MYLINK ]; then
    rm $MYLINK
    ln -s $MYDIR $MYLINK
else
    ln -s $MYDIR $MYLINK
fi

if [ -d $MYLINK ]; then
    echo '    ... OK'
else
    echo '    ... creation failed'
    return
fi

