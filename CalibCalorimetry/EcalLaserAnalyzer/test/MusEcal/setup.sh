#!/bin/sh
echo "setup in bash shell"
eval `scramv1 run -sh`
[[ ! ( -d lib ) ]] && mkdir lib
[[ ! ( -d bin ) ]] && mkdir bin
export MusEcal=${PWD}
export LD_LIBRARY_PATH=${ROOTSYS}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${MusEcal}/lib:${LD_LIBRARY_PATH}
export PATH=${MusEcal}/bin:${PATH}
