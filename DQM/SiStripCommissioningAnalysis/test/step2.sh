#!/bin/sh
#paramters: runnumber partition ishalow ishahigh ishastep vfslow vfshigh vfsstep

eval `scramv1 runtime -sh`
source /exports/slc4/development/FecSoftwareV3_0/config/oracle.env.bash afs
export ENV_TRACKER_DAQ=/exports/slc4/development/opt/trackerDAQ
export SCRATCH=`pwd`

root -l -q "step2.C($1,$3,$4,$5,$6,$7,$8)"

