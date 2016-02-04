#!/bin/sh


echo "Running $*"
PTYPE=`type -p $0`
PROG=`dirname ${PTYPE}`/../lib

if [ "${LD_LIBRARY_PATH}" == "" ]
then
    export LD_LIBRARY_PATH=${PROG}
else
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PROG}
fi

export LD_PRELOAD=libProfLogger.so

$*
