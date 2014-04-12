#!/bin/sh
# Symple script to set the include path for RooFit

cd $LOCALRT/src
includePath=`scram tool info roofitcore | grep INCLUDE | awk -F= '{print $2}'`
cd -
echo $includePath
cat RunCompareBias_template.C | sed "s:INCLUDEPATH:${includePath}:g" > RunCompareBias.C
root -l -b -q RunCompareBias.C
