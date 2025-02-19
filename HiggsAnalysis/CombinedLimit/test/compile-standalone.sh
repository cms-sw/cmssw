#!/bin/bash

## This script allows to compile the program as standalone, taking ROOT and BOOST from CERN AFS
## It also takes python 2.6 from /usr/bin/python2.6 in case your default python is 2.4 like in SLC5
## It is meant only for DEBUGGING and TESTING, and it is NOT supported.
## 
## Usage:
##  - if necessary, fix the paths to BOOST, GCC, ROOT
##  - go under HiggsAnalysis/CombinedLimit, and
##     - if necessary, clean up a previous build with
##         ./test/compile-standalone.sh clean
##     - if necessary, compile the program
##         ./test/compile-standalone.sh make
##     - setup your environment 
##         eval $(./test/compile-standalone.sh env)
##  - now you can run the program

if [[ "$1" == "" ]]; then echo "$0 (clean | make |env)"; exit 1; fi;

BASEDIR="$( cd "$( dirname "$0" )"/../../.. && pwd )" 
BOOST=/afs/cern.ch/cms/slc5_amd64_gcc434/external/boost/1.42.0-cms
MYGCC=/afs/cern.ch/sw/lcg/external/gcc/4.3.2/x86_64-slc5/setup.sh
MYROOT=/afs/cern.ch/sw/lcg/app/releases/ROOT/5.28.00/x86_64-slc5-gcc43-opt/root/bin/thisroot.sh
cd $BASEDIR/HiggsAnalysis/CombinedLimit;

if [[ "$1" == "env" ]]; then
    echo "source $MYGCC '' ;";
    echo "source $MYROOT;";
    echo "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:${BOOST}/lib\";";
    echo "export PATH=\"${BASEDIR}/HiggsAnalysis/CombinedLimit/tmp/bin:${PATH}\";";
    echo "export PYTHONPATH=\"${BASEDIR}/HiggsAnalysis/CombinedLimit/tmp/python:${PYTHONPATH}\";";
elif [[ "$1" == "clean" ]]; then
    test -f src/tmp_LinkDef.cc && rm src/tmp_LinkDef.cc
    test -f src/tmp_LinkDef.h  && rm src/tmp_LinkDef.h
    rm -r src/*.o tmp/ 2> /dev/null || /bin/true;
elif [[ "$1" == "make" ]]; then
    echo ">>> Setup "
    source $MYGCC  ''
    source $MYROOT 
    INC="-I${ROOTSYS}/include -I${BASEDIR} -I${BOOST}/include"
    
    echo ">>> Dictionaries "
    test -f src/tmp_LinkDef.cc && rm src/tmp_LinkDef.cc
    test -f src/tmp_LinkDef.h  && rm src/tmp_LinkDef.h
    echo rootcint -f src/tmp_LinkDef.cc -c -p ${INC} -fPIC src/LinkDef.h && \
         rootcint -f src/tmp_LinkDef.cc -c -p ${INC} -fPIC src/LinkDef.h || exit 1
    perl -i -npe 'print qq{#include "LinkDef.h"\n} if 1 .. 1' src/tmp_LinkDef.cc 

    echo ">>> Compile "
    CXXFLAGS="$(root-config --cflags) -O2  -I${BASEDIR} -I${BOOST}/include -fPIC ${CXXFLAGS}"
    CXXFLAGS=" -g ${CXXFLAGS}"
    for f in src/*.cc; do 
        test -f ${f/.cc/.o} && rm ${f/.cc/.o} 
        echo gcc ${CXXFLAGS} -c $f -o ${f/.cc/.o} && \
             gcc ${CXXFLAGS} -c $f -o ${f/.cc/.o} ;
    done
    test -f src/tmp_LinkDef.cc && rm src/tmp_LinkDef.cc
    test -f src/tmp_LinkDef.h  && rm src/tmp_LinkDef.h

    echo ">>> Link "
    mkdir -p tmp/bin
    LDFLAGS="-L${BOOST}/lib -lboost_program_options -lboost_system -lboost_filesystem ${LDFLAGS}"
    LDFLAGS="$(root-config --ldflags --libs)  -lRooFitCore -lRooStats -lRooFit -lFoam -lMinuit  ${LDFLAGS}"
    LDFLAGS="-lstdc++ -fPIC ${LDFLAGS}"
    echo gcc ${CXXFLAGS} src/*.o bin/combine.cpp ${LDFLAGS} -o tmp/bin/combine && 
         gcc ${CXXFLAGS} src/*.o bin/combine.cpp ${LDFLAGS} -o tmp/bin/combine || exit 3;

    echo ">>> Python"
    test -d tmp/bin || mkdir -p tmp/bin;
    test -d tmp/python/HiggsAnalysis || mkdir -p tmp/python/HiggsAnalysis;
    ln -sd $BASEDIR/HiggsAnalysis/CombinedLimit/python tmp/python/HiggsAnalysis/CombinedLimit
    touch tmp/python/HiggsAnalysis/__init__.py
    touch tmp/python/HiggsAnalysis/CombinedLimit/__init__.py

    # hack to get python 2.6
    if python -V 2>&1 | grep -q "Python 2.4"; then
        cp -s /usr/bin/python2.6 ${BASEDIR}/HiggsAnalysis/CombinedLimit/tmp/bin/python
    fi;
else
    echo "$0 (clean | make |env)"; 
    exit 1; 
fi;
