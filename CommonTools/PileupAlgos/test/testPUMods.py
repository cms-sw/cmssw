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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.source = cms.Source("PoolSource",
	fileNames  = cms.untracked.vstring(
		'/store/mc/RunIISpring15DR74/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/GEN-SIM-RECO/AsymptFlat0to50bx25Reco_MCRUN2_74_V9-v3/10000/0009D30B-0207-E511-B581-0026182FD753.root'
		)
)
process.source.inputCommands = cms.untracked.vstring("keep *",
                                                     "drop *_MEtoEDMConverter_*_*")

process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow     = cms.untracked.vstring('ProductNotFound'),
  fileMode    = cms.untracked.string('NOMERGE')
)

process.puppi.candName = 'packedPFCandidates'
process.puppi.candName = cms.InputTag('packedPFCandidates')
process.puppi.vertexName = cms.InputTag('offlineSlimmedPrimaryVertices')

process.packedPFCandidatesNoLep = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("abs(pdgId) != 13 && abs(pdgId) != 11"))
process.puppiNoLep = process.puppi.clone()
process.puppiNoLep.candName = cms.InputTag('packedPFCandidatesNoLep')
process.puppiNoLep.vertexName = cms.InputTag('offlineSlimmedPrimaryVertices')


process.puSequence = cms.Sequence(process.packedPFCandidatesNoLep+process.puppi+process.puppiNoLep)
process.p = cms.Path(process.puSequence)
process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = cms.untracked.vstring('keep *'),
                                  fileName       = cms.untracked.string ("Output.root")
)
# schedule definition                                                                                                       
process.outpath  = cms.EndPath(process.output) 
