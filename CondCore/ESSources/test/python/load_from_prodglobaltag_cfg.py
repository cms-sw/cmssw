#load all (and only!) records in a given globaltag

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('runNumber',
                 4294967294, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('runInterval',
                 0, #default just one run
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run used is runNunber+/-Interval; default 0")
options.register('globalTag',
                 'IDEAL', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "GlobalTag")
options.register('source',
                 'frontier', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "source: frontier, oracle, snapshot ")
options.register('record',
                 'ALL', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "record: either ALL or just one ")
options.register('overwrite',
                 '', #default none
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "record,tag,connectionstring")
options.register('add',
                 '', #default none
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "class,record,tag,connectionstring")

options.parseArguments()


import DLFCN, sys, os, time
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
rdbms = RDBMS("/afs/cern.ch/cms/DB/conddb")
logName = "oracle://cms_orcon_adg/CMS_COND_31X_POPCONLOG"
gdbName = "oracle://cms_orcon_adg/CMS_COND_31X_GLOBALTAG"
gName = options.globalTag+'::All'
globalTag = rdbms.globalTag(gdbName,gName,"","")



import FWCore.ParameterSet.Config as cms

records = cms.VPSet()
if ( options.record=="ALL") :
    for tag in globalTag.elements:
        records.append(
            cms.PSet(
            record = cms.string(tag.record),
            data = cms.vstring(tag.object)
            )
            )
else :
     for tag in globalTag.elements:
        if (tag.record==options.record) :
            records.append(
                cms.PSet(
                record = cms.string(tag.record),
                data = cms.vstring(tag.object)
                )
                )
            break

process = cms.Process("TEST")

process.add_(cms.Service("PrintEventSetupDataRetrieval", printProviders=cms.untracked.bool(True)))

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
process.GlobalTag.globaltag = gName

process.GlobalTag.RefreshEachRun=cms.untracked.bool(False)
process.GlobalTag.DumpStat=cms.untracked.bool(True)
process.GlobalTag.pfnPrefix=cms.untracked.string('')
process.GlobalTag.pfnPostfix=cms.untracked.string('')
#process.GlobalTag.DBParameters.authenticationPath = "/afs/cern.ch/cms/DB/conddb"
process.GlobalTag.DBParameters.connectionTimeOut = 0
process.GlobalTag.DBParameters.messageLevel = 0
process.GlobalTag.DBParameters.transactionId = cms.untracked.string("")


if(options.source=="oracle") :
  process.GlobalTag.pfnPrefix = cms.untracked.string('oracle://cms_orcon_adg/')

#process.GlobalTag.pfnPrefix = "frontier://FrontierArc/"
#process.GlobalTag.pfnPrefix = "oracle://cmsarc_lb/"
#process.GlobalTag.pfnPostfix = "_0912"
process.GlobalTag.toGet = cms.VPSet()

if (len(options.overwrite)>4):
  (orecord,otag,ocs) = options.overwrite.split(',')
  process.GlobalTag.toGet.append(
    cms.PSet(record = cms.string(orecord.strip()),
             tag = cms.string(otag.strip()),
             connect = cms.untracked.string(ocs.strip())
             )
    )

if (len(options.add)>5):
  (aclass, arecord,atag,acs) = options.add.split(',')
  process.GlobalTag.toGet.append(
    cms.PSet(record = cms.string(arecord.strip()),
             tag = cms.string(atag.strip()),
             connect = cms.untracked.string(acs.strip())
             )
    )
  records.append(
    cms.PSet(
    record = cms.string(arecord.strip()),
    data = cms.vstring(aclass.strip())
    )
    )



process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(options.runNumber+options.runInterval),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.runNumber-options.runInterval),
    interval = cms.uint64(1)
)

process.get = cms.EDFilter("EventSetupRecordDataGetter",
                           toGet = records,
                           verbose = cms.untracked.bool(True)
                           )

process.p = cms.Path(process.get)



