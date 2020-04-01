#!/bin/bash
set -e
set -x
CFLAGS="--std=c++17 -fPIC"
INCLUDE=(
  "-I/cvmfs/cms-ib.cern.ch/nweek-02622/slc7_amd64_gcc820/lcg/root/6.18.04-blocog/include" # ROOT
  "-I/cvmfs/cms-ib.cern.ch/nweek-02622/slc7_amd64_gcc820/external/boost/1.72.0/include" # BOOST
)
LIB=(
  "-L/cvmfs/cms-ib.cern.ch/week0/slc7_amd64_gcc820/cms/cmssw/CMSSW_11_1_PY3_X_2020-03-29-2300/external/slc7_amd64_gcc820/lib"
)
ROOTLIBS=(-lCore -lRIO -lNet -lHist -lMatrix -lThread -lTree -lMathCore -lTreePlayer -lGpad -lGraf3d -lGraf -lPhysics -lPostscript -lASImage)
OTHERLIBS=(-ldl -ljpeg -lpng)

# compile
g++ $CFLAGS "${INCLUDE[@]}" -c DQMRenderPlugin.cc
g++ $CFLAGS "${INCLUDE[@]}" -c render.cc

# link base renderplugin
g++ --shared "${LIB[@]}" "${ROOTLIBS[@]}" "${OTHERLIBS[@]}" DQMRenderPlugin.o -o renderplugin.so 

# TODO: link other renderplugins

# link executable
g++ "${LIB[@]}" "${ROOTLIBS[@]}" "${OTHERLIBS[@]}" $PWD/renderplugin.so render.o -o render

