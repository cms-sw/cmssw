import FWCore.ParameterSet.Config as cms

MuonTrackingRegionCommon = cms.PSet(
    MuonTrackingRegionBuilder = cms.PSet(
        DeltaEta = cms.double(0.2),
        DeltaPhi = cms.double(0.2),
        DeltaR = cms.double(0.2),
        DeltaZ = cms.double(15.9),
        EtaR_UpperLimit_Par1 = cms.double(0.25),
        EtaR_UpperLimit_Par2 = cms.double(0.15),
        Eta_fixed = cms.bool(False),
        Eta_min = cms.double(0.1),
        MeasurementTrackerName = cms.InputTag(""),
        OnDemand = cms.int32(-1),
        PhiR_UpperLimit_Par1 = cms.double(0.6),
        PhiR_UpperLimit_Par2 = cms.double(0.2),
        Phi_fixed = cms.bool(False),
        Phi_min = cms.double(0.1),
        Pt_fixed = cms.bool(False),
        Pt_min = cms.double(1.5),
        Rescale_Dz = cms.double(3.0),
        Rescale_eta = cms.double(3.0),
        Rescale_phi = cms.double(3.0),
        UseVertex = cms.bool(False),
        Z_fixed = cms.bool(True),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        input = cms.InputTag(""),
        maxRegions = cms.int32(1),
        precise = cms.bool(True),
        vertexCollection = cms.InputTag("")
    )
)