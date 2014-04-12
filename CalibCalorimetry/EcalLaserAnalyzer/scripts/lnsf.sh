#!/bin/bash

export MYDIR=$1
export MYLINK=$2
echo 'Creating link '$MYLINK

if [ -f $MYLINK ]; then
    rm $MYLINK
    ln -s $MYDIR $MYLINK
else
    ln -s $MYDIR $MYLINK
fi

if [ -f $MYLINK ]; then
    echo '    ... OK'
else
    echo '    ... creation failed'
    return
fi

