#!/bin/bash

export SUFFIX=$2
export CRABDIR=$1
export INPUTFILENAME=$3_*_*.root
export OUTPUTFILENAME=$3_${SUFFIX}.root

hadd rootfiles/${OUTPUTFILENAME} ${CRABDIR}/res/${INPUTFILENAME}
