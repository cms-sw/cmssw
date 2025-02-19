#!/bin/tcsh
eval `scramv1 runtime -csh`
SealPluginRefresh
# For Onine DB ( NOT Default)
#setenv TNS_ADMIN /afs/cern.ch/project/oracle/admin
#source /afs/cern.ch/project/oracle/script/setoraenv.csh -s 10201
# For Offline Condition DB 
setenv CORAL_AUTH_PATH /afs/cern.ch/cms/DB/conddb
cmsRun PedsMonWithDB.cfg



