import FWCore.ParameterSet.Config as cms

MuonTrackingRegionCommon = cms.PSet(
    MuonTrackingRegionBuilder = cms.PSet(
        # Upper limits on regions eta and phi size. Be careful changing these
        # you are changing the parametrization
        EtaR_UpperLimit_Par1 = cms.double(0.25),
        Eta_fixed = cms.double(0.2),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        Rescale_Dz = cms.double(3.0),
        Eta_min = cms.double(0.013),
        Rescale_phi = cms.double(3.0),
        DeltaR = cms.double(0.2),
        DeltaZ_Region = cms.double(15.9),
        Rescale_eta = cms.double(3.0),
        PhiR_UpperLimit_Par2 = cms.double(0.2),
        VertexCollection = cms.string('pixelVertices'),
        Phi_fixed = cms.double(0.2),
        EscapePt = cms.double(1.5),
        UseFixedRegion = cms.bool(False),
        PhiR_UpperLimit_Par1 = cms.double(0.6),
        EtaR_UpperLimit_Par2 = cms.double(0.15),
        Phi_min = cms.double(0.02),
        UseVertex = cms.bool(False)
    )
)

