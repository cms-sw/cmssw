import FWCore.ParameterSet.Config as cms

from MuonAnalysis.MuonAssociators.patMuonsWithTrigger_cff import *
muonMatchHLTL2.maxDeltaR = 0.3
muonMatchHLTL3.maxDeltaR = 0.1
patTriggerFull.l1GtReadoutRecordInputTag = cms.InputTag("gtDigis","","RECO")                 
patTrigger.collections.remove("hltL3MuonCandidates")
patTrigger.collections.append("hltHIL3MuonCandidates")
muonMatchHLTL3.matchedCuts = cms.string('coll("hltHIL3MuonCandidates")')

from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi import patPhotons
patPhotonSequence = cms.Sequence(patPhotons)

from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import patElectrons
patElectronSequence = cms.Sequence(patElectrons)

from HeavyIonsAnalysis.VectorBosonAnalysis.tupel_cfi import tupel
tupel.gjetSrc = cms.untracked.InputTag('ak4GenJets')
tupel.jetSrc  = cms.untracked.InputTag("ak4PFpatJetsWithBtagging")

tupelPatSequence = cms.Sequence(patMuonsWithTriggerSequence + patPhotonSequence + patElectronSequence + tupel)
