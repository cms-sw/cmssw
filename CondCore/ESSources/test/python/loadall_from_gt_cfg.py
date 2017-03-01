import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from CondCore.ESSources.PoolDBESSource_cfi import GlobalTag

options = VarParsing.VarParsing()
options.register('runNumber',
                 4294967294, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('globalTag',
                 'GR_P_V50', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "GlobalTag")
options.parseArguments()

import FWCore.ParameterSet.Config as cms


process = cms.Process("TEST")

#process.add_(cms.Service("PrintEventSetupDataRetrieval", printProviders=cms.untracked.bool(True)))

CondDBSetup = cms.PSet( DBParameters = cms.PSet(
                                                messageLevel = cms.untracked.int32(3),
                                                )
                        )

process.GlobalTag = cms.ESSource("PoolDBESSource",
                                 CondDBSetup,
                                 #connect = cms.string('oracle://cms_orcon_adg/CMS_CONDITIONS'),
                                 connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                 #    connect = cms.string('sqlite_fip:CondCore/TagCollection/data/GlobalTag.db'), #For use during release integration
                                 globaltag = cms.string(''),
                                 snapshotTime = cms.string('2012-01-20 23:59:59.000'),
                                 RefreshEachRun=cms.untracked.bool(False),
                                 DumpStat=cms.untracked.bool(False),
                                 pfnPrefix=cms.untracked.string(''),   
                                 pfnPostfix=cms.untracked.string('')
                                 )


process.GlobalTag.globaltag = options.globalTag
process.GlobalTag.DumpStat =  True
process.GlobalTag.toGet = cms.VPSet()
process.GlobalTag.toGet.append(
   cms.PSet(record = cms.string("BeamSpotObjectsRcd"),
            snapshotTime = cms.string('2014-01-01 00:00:00.000'),
           )
)



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
