import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('runNumber',
                 186234, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('globalTag',
                 'GR_P_V50', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "GlobalTag")
options.parseArguments()


process = cms.Process("TEST")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
process.GlobalTag.globaltag = options.globalTag
process.GlobalTag.RefreshEachRun=cms.untracked.bool(False)
process.GlobalTag.DumpStat=cms.untracked.bool(True)
process.GlobalTag.pfnPrefix=cms.untracked.string('')
process.GlobalTag.pfnPostfix=cms.untracked.string('')


process.GlobalTag.toGet = cms.VPSet()
process.GlobalTag.toGet.append(
   cms.PSet(record = cms.string("BeamSpotObjectsRcd"),
            tag = cms.string("firstcollisions"),
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
