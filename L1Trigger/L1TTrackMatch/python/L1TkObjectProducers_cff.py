import FWCore.ParameterSet.Config as cms

# Phase II EG seeds, Barrel (Crystal):

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkElectronsCrystal
pL1TkElectronsCrystal = cms.Path( L1TkElectronsCrystal )

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkElectronsEllipticMatchCrystal
pL1TkElectronsEllipticMatchCrystal = cms.Path( L1TkElectronsEllipticMatchCrystal )

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkElectronsLooseCrystal
pL1TkElectronsLooseCrystal = cms.Path( L1TkElectronsLooseCrystal )

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkIsoElectronsCrystal
pL1TkIsoElectronsCrystal = cms.Path( L1TkIsoElectronsCrystal )

from L1Trigger.L1TTrackMatch.L1TkEmParticleProducer_cfi import L1TkPhotonsCrystal
pL1TkPhotonsCrystal = cms.Path( L1TkPhotonsCrystal )

#+ from L1Trigger.L1TTrackMatch.L1WP2ElectronProducer_cfi import L1WP2Electrons
#+ pL1WP2Electrons = cms.Path( L1WP2Electrons)

# Phase II EG seeds, Endcap (HGC):
# Two objects for now to follow 2017 discussions, merging collections would be nice...

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkElectronsHGC
pL1TkElectronsHGC = cms.Path( L1TkElectronsHGC )

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkElectronsEllipticMatchHGC
pL1TkElectronsEllipticMatchHGC = cms.Path( L1TkElectronsEllipticMatchHGC )

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkIsoElectronsHGC
pL1TkIsoElectronsHGC = cms.Path( L1TkIsoElectronsHGC )

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkElectronsLooseHGC
pL1TkElectronsLooseHGC = cms.Path( L1TkElectronsLooseHGC )

from L1Trigger.L1TTrackMatch.L1TkEmParticleProducer_cfi import L1TkPhotonsHGC
pL1TkPhotonsHGC = cms.Path( L1TkPhotonsHGC )


#Other tk Objects

# from L1Trigger.L1TTrackMatch.L1TrackerJetProducer_cfi import L1TrackerJets
# pL1TrackerJets = cms.Path( L1TrackerJets)

# from L1Trigger.TwoLayerJets.TwoLayerJets_cfi import TwoLayerJets
# pL1TwoLayerJets = cms.Path( TwoLayerJets)

# from L1Trigger.L1TTrackMatch.L1TkCaloJetProducer_cfi import L1TkCaloJets
# pL1TkCaloJets = cms.Path( L1TkCaloJets)

from L1Trigger.VertexFinder.l1tVertexProducer_cfi import l1tVertexProducer
pVertexProducer = cms.Path( l1tVertexProducer )

# from L1Trigger.L1TTrackMatch.l1tTrackerEtMiss_cfi import l1tTrackerEtMiss
# pL1TrkMET = cms.Path( l1tTrackerEtMiss )

# from L1Trigger.L1TTrackMatch.l1tTrackerHTMiss_cfi import l1tTkCaloHTMissVtx, l1tTrackerHTMiss
# pL1TkCaloHTMissVtx = cms.Path( l1tTkCaloHTMissVtx )
# pL1TrackerHTMiss = cms.Path( l1tTrackerHTMiss )

from L1Trigger.L1TTrackMatch.L1TkMuonProducer_cfi import L1TkMuons, L1TkMuonsTP
pL1TkMuon = cms.Path( L1TkMuons * L1TkMuonsTP )

from L1Trigger.L1TTrackMatch.L1TkGlbMuonProducer_cfi import L1TkGlbMuons
pL1TkGlbMuon = cms.Path( L1TkGlbMuons )

# from L1Trigger.L1TTrackMatch.L1TrkTauFromCaloProducer_cfi import L1TrkTauFromCalo
# pL1TrkTauFromCalo = cms.Path( L1TrkTauFromCalo )

# from L1Trigger.Phase2L1Taus.TkTauProducer_cfi import L1TrkTaus
# L1TrackerTaus = L1TrkTaus.clone()

# from L1Trigger.Phase2L1Taus.TkEGTauProducer_cfi import L1TkEGTaus

# from L1Trigger.Phase2L1Taus.L1CaloTkTauProducer_cfi import L1CaloTkTaus
# L1TkCaloTaus = L1CaloTkTaus.clone()
