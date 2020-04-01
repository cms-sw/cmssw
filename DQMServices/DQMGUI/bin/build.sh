#!/bin/bash
set -e
set -x
CFLAGS="--std=c++17 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
INCLUDE=(
  "-I/cvmfs/cms-ib.cern.ch/nweek-02622/slc7_amd64_gcc820/lcg/root/6.18.04-blocog/include" # ROOT
  "-I/cvmfs/cms-ib.cern.ch/nweek-02622/slc7_amd64_gcc820/external/boost/1.72.0/include" # BOOST
)
LIB=(
  "-L/cvmfs/cms-ib.cern.ch/week0/slc7_amd64_gcc820/cms/cmssw/CMSSW_11_1_PY3_X_2020-03-29-2300/external/slc7_amd64_gcc820/lib"
)
ROOTLIBS=(-lCore -lRIO -lNet -lHist -lMatrix -lThread -lTree -lMathCore -lTreePlayer -lGpad -lGraf3d -lGraf -lPhysics -lPostscript -lASImage)
OTHERLIBS=(-ldl -ljpeg -lpng "-lstdc++fs")
# I have no idea why G++ does not read symbols from here with -lstdc++fs. I got that filename out of strace, so it was sure opened...
HACKS="/cvmfs/cms-ib.cern.ch/nweek-02622/slc7_amd64_gcc820/external/gcc/8.2.0-pafccj/bin/../lib/gcc/x86_64-unknown-linux-gnu/8.3.1/../../../../lib64/libstdc++fs.a" 

# compile
g++ $CFLAGS "${INCLUDE[@]}" -c DQMRenderPlugin.cc
g++ $CFLAGS "${INCLUDE[@]}" -c render.cc

# link base renderplugin
g++ --shared "${LIB[@]}" "${ROOTLIBS[@]}" "${OTHERLIBS[@]}" DQMRenderPlugin.o -o renderplugin.so 

# TODO: link other renderplugins

# link executable
g++ $CFLAGS "${LIB[@]}" "${ROOTLIBS[@]}" "${OTHERLIBS[@]}" $PWD/renderplugin.so render.o -o render $HACKS

