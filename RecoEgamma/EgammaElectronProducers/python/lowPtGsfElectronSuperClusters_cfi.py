
import FWCore.ParameterSet.Config as cms

lowPtGsfElectronSuperClusters = cms.EDProducer(
    "LowPtGsfElectronSCProducer",
    gsfPfRecTracks = cms.InputTag("pfTrackElec"),
    pfRecTracks = cms.InputTag("pfTrackOpen"), #@@ not needed, debug only 
    gsfTracks = cms.InputTag("electronGsfTracksOpen"), #@@ not needed, debug only 
    ecalClusters = cms.InputTag("particleFlowClusterECAL"),
    hcalClusters = cms.InputTag("particleFlowClusterHCAL"),
)
