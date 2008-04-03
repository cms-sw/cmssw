#!/bin/sh

#paramters: runnumber ishalow ishahigh ishastep vfslow vfshigh vfsstep

eval `scramv1 runtime -sh`
source /exports/slc4/development/FecSoftwareV3_0/config/oracle.env.bash afs
export ENV_TRACKER_DAQ=/exports/slc4/development/opt/trackerDAQ
export SCRATCH=`pwd`

for isha in `seq $2 $4 $3`; do
for vfs  in `seq $5 $7 $6`; do

#put the right files in /input
rm -f input/*
cp *${1}_*_ISHA${isha}_VFS${vfs}_*.root input/

#prepare input
cp OfflineDbClient.template OfflineClientScan.cfg
ex OfflineClientScan.cfg +":%s/RUNNUMBER/$1" +wq

#run
cmsRun OfflineClientScan.cfg

#rename result
mv SiStripCommissioningClient_000${1}.root SiStripCommissioningClient_000${1}_ISHA${isha}_VFS${vfs}.root

done
done

