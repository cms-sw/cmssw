import FWCore.ParameterSet.Config as cms

process = cms.Process('TestPUMods')
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.MessageLogger.cerr.FwkReport.reportEvery = 10
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.load('CommonTools/PileupAlgos/Puppi_cff')
process.load('CommonTools/PileupAlgos/softKiller_cfi')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(50) )
process.source = cms.Source("PoolSource",
                            fileNames  = cms.untracked.vstring('/store/relval/CMSSW_7_5_0_pre4/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/MCRUN2_75_V1-v1/00000/328B19C2-4DF6-E411-92C1-003048FFD796.root')
)
process.source.inputCommands = cms.untracked.vstring("keep *",
                                                     "drop *_MEtoEDMConverter_*_*")

process.puppi.candName = cms.InputTag('packedPFCandidates')
process.puppi.vertexName = cms.InputTag('offlineSlimmedPrimaryVertices')
process.softKiller.PFCandidates = cms.InputTag('packedPFCandidates')

process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow     = cms.untracked.vstring('ProductNotFound'),
  fileMode    = cms.untracked.string('NOMERGE')
)


process.puSequence = cms.Sequence(process.puppi*process.softKiller)
#process.puSequence = cms.Sequence(process.puppi)
process.p = cms.Path(process.puSequence)
process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = cms.untracked.vstring('drop *',
                                                                         'keep *_particleFlow_*_*',
                                                                         'keep *_*_*_TestPUMods'),
                                  fileName       = cms.untracked.string ("Output.root")
)
# schedule definition                                                                                                       
process.outpath  = cms.EndPath(process.output) 
