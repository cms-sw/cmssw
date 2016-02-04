import FWCore.ParameterSet.Config as cms
process = cms.Process('TestPUMods')

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Geometry_cff")
process.MessageLogger.cerr.FwkReport.reportEvery = 10
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.load('CommonTools/PileupAlgos/Puppi_cff')
process.load('CommonTools/PileupAlgos/PhotonPuppi_cff')
from CommonTools.PileupAlgos.PhotonPuppi_cff import setupPuppiPhoton

process.load('CommonTools/PileupAlgos/softKiller_cfi')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
process.source = cms.Source("PoolSource",
	fileNames  = cms.untracked.vstring(
		'/store/mc/RunIISpring15MiniAODv2/GJets_HT-600ToInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2-v1/60000/E2B9A332-966F-E511-9BAA-0026B927865E.root'
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

process.load('RecoMET.METProducers.PFMET_cfi')
process.pfMet.src = cms.InputTag('puppiPhoton')
setupPuppiPhoton(process)
process.puppiPhoton.puppiCandName    = 'puppiNoLep'

process.puSequence = cms.Sequence(process.packedPFCandidatesNoLep*process.puppi*process.puppiNoLep*process.egmPhotonIDSequence*process.puppiPhoton*process.pfMet)
process.p = cms.Path(process.puSequence)
process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = cms.untracked.vstring('keep *'),
                                  fileName       = cms.untracked.string ("Output.root")
)
# schedule definition                                                                                                       
process.outpath  = cms.EndPath(process.output) 
