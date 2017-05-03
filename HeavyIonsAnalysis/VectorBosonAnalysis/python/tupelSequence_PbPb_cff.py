import FWCore.ParameterSet.Config as cms

from MuonAnalysis.MuonAssociators.patMuonsWithTrigger_cff import *
muonMatchHLTL2.maxDeltaR = 0.3
muonMatchHLTL3.maxDeltaR = 0.1
patTriggerFull.l1GtReadoutRecordInputTag = cms.InputTag("gtDigis","","RECO")                 
patTrigger.collections.remove("hltL3MuonCandidates")
patTrigger.collections.append("hltHIL3MuonCandidates")
muonMatchHLTL3.matchedCuts = cms.string('coll("hltHIL3MuonCandidates")')
patMuonsWithoutTrigger.pvSrc = cms.InputTag("offlinePrimaryVertex")

from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi import patPhotons
patPhotons.photonSource = cms.InputTag("photons") # PbPb GED photons are not trustworthy (yet)
patPhotons.electronSource = cms.InputTag("gedGsfElectronsTmp")
patPhotons.embedRecHits = cms.bool(False)
patPhotons.addPhotonID = cms.bool(False)
patPhotonSequence = cms.Sequence(patPhotons)

from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import patElectrons
patElectrons.electronSource = cms.InputTag("gedGsfElectronsTmp")
patElectrons.reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
patElectrons.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
patElectrons.pvSrc = cms.InputTag("offlinePrimaryVertex")
patElectrons.addElectronID = cms.bool(False)
patElectronSequence = cms.Sequence(patElectrons)

from HeavyIonsAnalysis.VectorBosonAnalysis.tupel_cfi import tupel

tupelPatSequence = cms.Sequence(patMuonsWithTriggerSequence + patPhotonSequence + patElectronSequence + tupel)
