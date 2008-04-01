#!/bin/sh

#paramters: runnumber ishalow ishahigh ishastep vfslow vfshigh vfsstep

for isha in `seq $2 $4 $3`; do
for vfs  in `seq $5 $7 $6`; do

#put the right files in /input
rm -f input/*
cp *${1}_*_ISHA${isha}_VFS${vfs}_*.root input/

#prepare input
cp OfflineDbClient.template OfflineClientScan.cfg
ex OfflineClientScan.cfg +":%s/RUNNUMBER/$1" +wq

#run
eval `scramv1 runtime -sh`
cmsRun OfflineClientScan.cfg

#rename result
mv SiStripCommissioningClient_00039385.root SiStripCommissioningClient_00039385_ISHA${isha}_VFS${vfs}.root

done
done

