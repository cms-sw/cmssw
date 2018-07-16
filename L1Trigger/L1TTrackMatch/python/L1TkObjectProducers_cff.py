import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkElectrons
pL1TkElectrons = cms.Path( L1TkElectrons )

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkIsoElectrons
pL1TkIsoElectrons = cms.Path( L1TkIsoElectrons )

from L1Trigger.L1TTrackMatch.L1TkEmParticleProducer_cfi import L1TkPhotons
pL1TkPhotons = cms.Path( L1TkPhotons )

from L1Trigger.L1TTrackMatch.L1TrackerJetProducer_cfi import L1TrackerJets
pL1TrackerJets = cms.Path( L1TrackerJets)

from L1Trigger.L1TTrackMatch.L1TkCaloJetProducer_cfi import L1TkCaloJets
pL1TkCaloJets = cms.Path( L1TkCaloJets)

from L1Trigger.L1TTrackMatch.L1TkPrimaryVertexProducer_cfi import L1TkPrimaryVertex
pL1TkPrimaryVertex = cms.Path( L1TkPrimaryVertex )

from L1Trigger.L1TTrackMatch.L1TrackerEtMissProducer_cfi import L1TrackerEtMiss
pL1TrkMET = cms.Path( L1TrackerEtMiss )

from L1Trigger.L1TTrackMatch.L1TkHTMissProducer_cfi import L1TkCaloHTMissVtx, L1TrackerHTMiss
pL1TkCaloHTMissVtx = cms.Path( L1TkCaloHTMissVtx )
pL1TrackerHTMiss = cms.Path( L1TrackerHTMiss )

from L1Trigger.L1TTrackMatch.L1TkMuonProducer_cfi import L1TkMuons
pL1TkMuon = cms.Path( L1TkMuons )

from L1Trigger.L1TTrackMatch.L1TkGlbMuonProducer_cfi import L1TkGlbMuons
pL1TkGlbMuon = cms.Path( L1TkGlbMuons )

from L1Trigger.L1TTrackMatch.L1TkTauFromCaloProducer_cfi import L1TkTauFromCalo
pL1TkTauFromCalo = cms.Path( L1TkTauFromCalo )
