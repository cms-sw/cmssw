#!/bin/sh

. /opt/cmssw/cmsset_default.sh
eval `scram --arch slc6_amd64_gcc462 runtime -sh`

dir=`dirname $0`

cmsRun ${dir}/startFU.py
