import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkElectrons
pL1TkElectrons = cms.Path( L1TkElectrons )

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkIsoElectrons
pL1TkIsoElectrons = cms.Path( L1TkIsoElectrons )

from L1Trigger.L1TTrackMatch.L1TkEmParticleProducer_cfi import L1TkPhotons 
pL1TkPhotons = cms.Path( L1TkPhotons )

from L1Trigger.L1TTrackMatch.L1TkJetProducer_cfi import L1TkJets
pL1TkJets = cms.Path( L1TkJets)

from L1Trigger.L1TTrackMatch.L1TkPrimaryVertexProducer_cfi import L1TkPrimaryVertex
pL1TkPrimaryVertex = cms.Path( L1TkPrimaryVertex )

from L1Trigger.L1TTrackMatch.L1TkEtMissProducer_cfi import L1TkEtMiss
pL1TrkMET = cms.Path( L1TkEtMiss )

from L1Trigger.L1TTrackMatch.L1TkHTMissProducer_cfi import L1TkHTMissVtx
pL1TkHTMissVtx = cms.Path( L1TkHTMissVtx )

from L1Trigger.L1TTrackMatch.L1TkMuonProducer_cfi import L1TkMuons
pL1TkMuon = cms.Path( L1TkMuons )

from L1Trigger.L1TTrackMatch.L1TkTauFromCaloProducer_cfi import L1TkTauFromCalo
pL1TkTauFromCalo = cms.Path( L1TkTauFromCalo )
