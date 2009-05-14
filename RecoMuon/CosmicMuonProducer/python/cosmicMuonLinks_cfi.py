import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

cosmicMuonLinks = cms.EDProducer("CosmicMuonLinksProducer",
    MuonServiceProxy,
    Maps = cms.VPSet(
     cms.PSet(
       subTrack = cms.InputTag("cosmicMuonsBarrelOnly"),
       parentTrack = cms.InputTag("cosmicMuons") 
     ),
     cms.PSet(
       subTrack = cms.InputTag("cosmicMuonsEndCapsOnly"),
       parentTrack = cms.InputTag("cosmicMuons")
     ),
     cms.PSet(
       subTrack = cms.InputTag("cosmicMuons"),
       parentTrack = cms.InputTag("cosmicMuons1LegBarrelOnly")
     ),
     cms.PSet(
       subTrack = cms.InputTag("cosmicMuonsNoDriftBarrelOnly"),
       parentTrack = cms.InputTag("cosmicMuons1LegBarrelOnly")
     )
   )
)

