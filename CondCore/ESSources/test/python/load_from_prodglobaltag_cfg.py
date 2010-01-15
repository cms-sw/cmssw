import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('runNumber',
                 4294967294, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('globalTag',
                 'IDEAL', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "GlobalTag")
options.parseArguments()


import DLFCN, sys, os, time
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
rdbms = RDBMS("/afs/cern.ch/cms/DB/conddb")
logName = "oracle://cms_orcoff_prod/CMS_COND_31X_POPCONLOG"
gdbName = "oracle://cms_orcoff_prod/CMS_COND_31X_GLOBALTAG"
gName = options.globalTag+'::All'
globalTag = rdbms.globalTag(gdbName,gName,"","")


import FWCore.ParameterSet.Config as cms

records = cms.VPSet()
for tag in globalTag.elements:
    records.append(
      cms.PSet(
        record = cms.string(tag.record),
        data = cms.vstring(tag.object)
        )
      )


process = cms.Process("TEST")

process.add_(cms.Service("PrintEventSetupDataRetrieval", printProviders=cms.untracked.bool(True)))

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
process.GlobalTag.globaltag = gName
process.GlobalTag.RefreshEachRun=cms.untracked.bool(False)
process.GlobalTag.DumpStat=cms.untracked.bool(True)
process.GlobalTag.pfnPrefix=cms.untracked.string('')
process.GlobalTag.pfnPostfix=cms.untracked.string('')
process.GlobalTag.DBParameters.authenticationPath = "/afs/cern.ch/cms/DB/conddb"
process.GlobalTag.DBParameters.connectionTimeOut = 0
process.GlobalTag.DBParameters.messageLevel = 0
process.GlobalTag.DBParameters.transactionId = cms.untracked.string("")



# 'GR09_P_V6::All'
#'CRAFT09_R_V9::All'
#'MC_31X_V9::All'
#'GR09_31X_V5P::All'
#process.GlobalTag.pfnPrefix = "frontier://FrontierArc/"
#process.GlobalTag.pfnPostfix = "_0911"
#process.GlobalTag.toGet = cms.VPSet()
#process.GlobalTag.toGet.append(
#   cms.PSet(record = cms.string("BeamSpotObjectsRcd"),
#            tag = cms.string("firstcollisions"),
#             connect = cms.untracked.string("frontier://PromptProd/CMS_COND_31X_BEAMSPOT")
#           )
#)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(options.runNumber+1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.runNumber-1),
    interval = cms.uint64(1)
)

process.get = cms.EDFilter("EventSetupRecordDataGetter",
                           toGet = records,
                           verbose = cms.untracked.bool(True)
                           )

process.p = cms.Path(process.get)



