#!/bin/sh

#paramters: runnumber partition ishalow ishahigh ishastep vfslow vfshigh vfsstep

eval `scramv1 runtime -sh`
source /exports/slc4/development/FecSoftwareV3_0/config/oracle.env.bash afs
export ENV_TRACKER_DAQ=/exports/slc4/development/opt/trackerDAQ
export SCRATCH=`pwd`
cp summaryScan.xml ../../..
mkdir input

for isha in `seq $3 $5 $4`; do
for vfs  in `seq $6 $8 $7`; do

#put the right files in /input
rm -f input/*
cp /tmp/*${1}_*_ISHA${isha}_VFS${vfs}_*.root input/

#prepare input
cp OfflineDbClient.template OfflineClientScan.cfg
ex OfflineClientScan.cfg +":%s/RUNNUMBER/$1" +wq
ex OfflineClientScan.cfg +":%s/PARTITION/$2" +wq

#run
cmsRun OfflineClientScan.cfg

#rename result
mv SiStripCommissioningClient_000${1}.root SiStripCommissioningClient_000${1}_ISHA${isha}_VFS${vfs}.root

done
done

