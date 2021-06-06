
import FWCore.ParameterSet.Config as cms
process = cms.Process("HLTPSCheck")

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('analysis') 
options.register('globalTag','auto:run2_data',options.multiplicity.singleton,options.varType.string,"global tag to use")
options.parseArguments()

print options.inputFiles
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(options.inputFiles),  
                          )

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(5000),
    limit = cms.untracked.int32(10000000)
)

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)


process.hltPSExample = cms.EDAnalyzer("HLTPrescaleExample",
                                      hltProcess=cms.string("HLT"),
#                                      hltPath=cms.string("HLT_Photon50_v13"),
                                      hltPath=cms.string("HLT_Photon33_v5"),                         
                                      hltPSProvCfg=cms.PSet(
                                          stageL1Trigger = cms.uint32(2)
                                      )
                                  )


process.p = cms.Path(
    process.hltPSExample
)



print "global tag: ",process.GlobalTag.globaltag



