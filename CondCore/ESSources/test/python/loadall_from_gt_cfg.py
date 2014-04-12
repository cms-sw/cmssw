import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('runNumber',
                 4294967294, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('globalTag',
                 'GR_P_V32', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "GlobalTag")
options.parseArguments()

import FWCore.ParameterSet.Config as cms


process = cms.Process("TEST")

process.add_(cms.Service("PrintEventSetupDataRetrieval", printProviders=cms.untracked.bool(True)))

CondDBSetup = cms.PSet( DBParameters = cms.PSet(authenticationPath = cms.untracked.string('.'),
                                                connectionRetrialPeriod = cms.untracked.int32(10),
                                                idleConnectionCleanupPeriod = cms.untracked.int32(10),
                                                messageLevel = cms.untracked.int32(0),
                                                enablePoolAutomaticCleanUp = cms.untracked.bool(False),
                                                enableConnectionSharing = cms.untracked.bool(True),
                                                connectionRetrialTimeOut = cms.untracked.int32(60),
                                                connectionTimeOut = cms.untracked.int32(0),
                                                enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False)
                                                )
                        )

process.GlobalTag = cms.ESSource("PoolDBESSource",
                                 CondDBSetup,
                                 connect = cms.string('frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'),
                                 #    connect = cms.string('sqlite_fip:CondCore/TagCollection/data/GlobalTag.db'), #For use during release integration
                                 globaltag = cms.string('UNSPECIFIED::All'),
                                 RefreshEachRun=cms.untracked.bool(False),
                                 DumpStat=cms.untracked.bool(False),
                                 pfnPrefix=cms.untracked.string(''),   
                                 pfnPostfix=cms.untracked.string('')
                                 )


process.GlobalTag.globaltag = options.globalTag+'::All'
process.GlobalTag.DumpStat =  True
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


process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
                             toGet =  cms.VPSet(),
                             verbose = cms.untracked.bool(True)
                             )

process.p = cms.Path(process.get)
