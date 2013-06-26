#!/bin/tcsh

echo "Running ECAL Laser Correction Service Test"

source /etc/hepix/oracle_env.csh
setenv TNS_ADMIN /afs/cern.ch/project/oracle/admin
 
cmsRun test_ecallaserdb_2.cfg

