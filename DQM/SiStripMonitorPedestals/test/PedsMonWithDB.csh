#!/bin/tcsh
eval `scramv1 runtime -csh`
SealPluginRefresh
source /afs/cern.ch/project/oracle/script/setoraenv.csh -s 10201
cmsRun PedsMonWithDB.cfg



