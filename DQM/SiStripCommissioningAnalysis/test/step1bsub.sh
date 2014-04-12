#!/bin/sh
trap -p "echo got a signal" SIGABRT SIGBUS SIGILL SIGINT SIGKILL SIGQUIT SIGSEGV SIGSTOP 
echo preparing environment
WORKDIR=`pwd`
cd $5
eval `scramv1 runtime -sh`
cd $WORKDIR
export SCRATCH=`pwd`
cp $5/OfflineDbClient.template OfflineClientScan.cfg
ex OfflineClientScan.cfg +":%s/RUNNUMBER/$1" +wq
ex OfflineClientScan.cfg +":%s/PARTITION/$2" +wq
echo checking out data
for i in `nsls /castor/cern.ch/user/d/delaer/CMStracker/${1} | grep ISHA${3}_VFS${4}`; { rfcp /castor/cern.ch/user/d/delaer/CMStracker/${1}/$i .; }
echo running in $SCRATCH
cmsRun OfflineClientScan.cfg || echo 
echo saving client file as /castor/cern.ch/user/d/delaer/CMStracker/${1}/SiStripCommissioningClient_000${1}_ISHA${3}_VFS${4}.root
rfcp SiStripCommissioningClient_000${1}.root /castor/cern.ch/user/d/delaer/CMStracker/${1}/SiStripCommissioningClient_000${1}_ISHA${3}_VFS${4}.root
