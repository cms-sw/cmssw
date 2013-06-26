#!/bin/bash

if [ $# -lt 3 ]
then
    echo "usage $0 function_id cutoff fileset_prefix <n>"
    echo " cutoff = 0 includes everything"
    echo " edit Parameters.py to change the stack up/down values"
    echo " <n> is optional, if there, function names are used instead of IDs"
	exit -1
fi

fid=$1
cutoff=$2
prefix=$3
USE_NAMES="n"

if [ $# -gt 3 ]
then
    USE_NAMES="y"
fi

PTYPE=`type -p $0`

PROG_DIR=`dirname ${PTYPE}`
PROG_LIB=${PROG_DIR}/../lib

if [ "${LD_LIBRARY_PATH}" == "" ]
then
    export LD_LIBRARY_PATH=${PROG_LIB}
else
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PROG_LIB}
fi

export PATH=${PATH}:${PROG_DIR}:

echo "function ID = ${fid}"
echo "path cutoff = ${cutoff}"
echo "file prefix = ${prefix}"

echo "selecting the function"
NewTree.py $fid ${prefix}paths
echo "converting to edges"
TreeToEdges.py ${fid}_paths ${fid}_edges ${cutoff}
echo "converting to graphviz"

if [ "${USE_NAMES}" == "y" ]
then
    echo "running with names"
    EdgesToViz.py ${fid}_edges ${fid}_dig ${prefix}names y
else
    EdgesToViz.py ${fid}_edges ${fid}_dig ${prefix}names
fi

echo "plotting ps"
dot -Tps -Gsize="8.5,11" -o ${fid}.ps ${fid}_dig

echo "plotting png"
dot -Tpng -o ${fid}.png ${fid}_dig

