#!/bin/csh
set INCLUDE=`scram tool info onnxruntime | grep ^INCLUDE= | cut -d= -f2`
set LIBDIR=`scram tool info onnxruntime | grep ^LIBDIR= | cut -d= -f2`

g++ -Wall -Wno-deprecated \
    -I./ \
    -I${INCLUDE} \
    `root-config --cflags` \
    CalibMain.C \
    -L${LIBDIR} \
    -lonnxruntime \
    -o CalibMain.exe \
    `root-config --libs`

