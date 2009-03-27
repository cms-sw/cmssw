import FWCore.ParameterSet.Config as cms

muonTCMETValueMapProducer = cms.EDProducer("MuonTCMETValueMapProducer",
     muonInputTag     = cms.InputTag("muons"),
     beamSpotInputTag = cms.InputTag("offlineBeamSpot"),
     pt_min           = cms.double(2.),
     pt_max           = cms.double(100.),
     eta_max          = cms.double(2.4),
     chi2_max         = cms.double(4),
     nhits_min        = cms.double(11),
     d0_max           = cms.double(0.1),   
     d0_muon          = cms.double(0.2),
     pt_muon          = cms.double(10),
     eta_muon         = cms.double(2.4),
     chi2_muon        = cms.double(10),
     nhits_muon       = cms.double(11),
     global_muon      = cms.bool(True),
     tracker_muon     = cms.bool(True),
     qoverpError_muon = cms.double(999999.9),
     deltaPt_muon     = cms.double(0.2)
)
