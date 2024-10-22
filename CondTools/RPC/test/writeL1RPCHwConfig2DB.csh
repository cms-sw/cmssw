#!/bin/csh
#
set goon = 1
set period = 1800
#
while ($goon == 1)
cat >! inspectRunInfo.py << %
import os,sys, DLFCN
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
rdbms = RDBMS("/afs/cern.ch/cms/DB/conddb")
#rdbms = RDBMS("/nfshome0/xiezhen/conddb")

dbName =  "oracle://cms_orcoff_prod/CMS_COND_21X_RUN_INFO"
logName = "oracle://cms_orcoff_prod/CMS_COND_21X_POPCONLOG"
#dbName =  "oracle://cms_orcon_prod/CMS_COND_21X_RUN_INFO"
#logName = "oracle://cms_orcon_prod/CMS_COND_21X_POPCONLOG"

rdbms.setLogger(logName)
from CondCore.Utilities import iovInspector as inspect

db = rdbms.getDB(dbName)
tags = db.allTags()

# for inspecting last run after run has started  
tag = 'runinfostart_test'

# for inspecting last run after run has stopped  
#tag = 'runinfo_test'

try :
    log = db.lastLogEntry(tag)
    iov = inspect.Iov(db,tag,0,0,0,1)
    for x in  iov.summaries():
        print x[1]
except RuntimeError :
    print "0"
%
#
set lastrun = `python inspectRunInfo.py`
#
cat >! writeL1RPCHwConfig2DB.cfg << %%
process Write2DB = {

  service = MessageLogger {
     untracked vstring destinations = {"cout"}
     untracked PSet cout = 
     {
       untracked  PSet default = { untracked int32 limit = 0 }
     }
  }

  source = EmptyIOVSource {
    string timetype = "runnumber"
    uint64 firstValue = $lastrun
    uint64 lastValue = $lastrun
    uint64 interval = 1
  }

  include "CondCore/DBCommon/data/CondDBCommon.cfi"

# Select output destination:
## to write into sqlite file
  replace CondDBCommon.connect = "sqlite_file:L1RPCHwConfig.db"
#  replace CondDBCommon.DBParameters.authenticationPath="."
#
## to write into int2r_orcoff
#  replace CondDBCommon.connect = "oracle://cms_orcoff_int2r/CMS_COND_RPC"
#  replace CondDBCommon.DBParameters.authenticationPath="."    
#
## to write into orcon
#  replace CondDBCommon.connect = "oracle://cms_orcon_prod/CMS_COND_21X_RPC"
#  replace CondDBCommon.DBParameters.authenticationPath="/afs/cern.ch/cms/DB/conddb"

  service = PoolDBOutputService {
    using CondDBCommon
    VPSet toPut = {
      { string record = "L1RPCHwConfigRcd" string tag = "L1RPCHwConfig_v1"}
    }
    untracked string logconnect = "sqlite_file:L1RPCHwConfig_log.db"
  }

  module WriteInDB = L1RPCHwConfigDBWriter {
    string record = "L1RPCHwConfigRcd"
    untracked bool loggingOn = true
    bool SinceAppendMode = true
    PSet Source = {
      untracked int32 WriteDummy = 0
      untracked int32 Validate = 1
      untracked int32 FirstBX = 0
      untracked int32 LastBX = 0
      untracked string OnlineConn = "oracle://CMS_OMDS_LB/CMS_RPC_CONF"
      untracked string OnlineAuthPath = "."
    }
  }

  path p = {WriteInDB}
}
%%
#
cmsRun writeL1RPCHwConfig2DB.cfg
#
if ( -f writeL1RPCHwConfig2DB.inp )then
 set goon = `cat writeL1RPCHwConfig2DB.inp`
endif
#
sleep $period
end
#
