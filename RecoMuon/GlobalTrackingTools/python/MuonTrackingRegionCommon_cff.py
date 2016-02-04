import FWCore.ParameterSet.Config as cms

MuonTrackingRegionCommon = cms.PSet(
    MuonTrackingRegionBuilder = cms.PSet(
        # Upper limits on regions eta and phi size. Be careful changing these
        # you are changing the parametrization
        EtaR_UpperLimit_Par1 = cms.double(0.25),
        Eta_fixed = cms.double(0.2),
        # -1. : nothing is made on demand
        # 0.0 : strip only are made on demand
        # 1.0 : strip and pixel are made on demand
        OnDemand = cms.double(-1.0),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        Rescale_Dz = cms.double(3.0), ## Delta Z from tracker

        Eta_min = cms.double(0.1), ## Lower limit to the region eta

        Rescale_phi = cms.double(3.0), ## Rescale Error in Phi

        DeltaR = cms.double(0.2), ## geometrical constraint

        DeltaZ_Region = cms.double(15.9), ## half interaction region

        Rescale_eta = cms.double(3.0), ## Rescale Error in Eta

        PhiR_UpperLimit_Par2 = cms.double(0.2),
        vertexCollection = cms.InputTag("pixelVertices"),
        Phi_fixed = cms.double(0.2),
        EscapePt = cms.double(1.5), ## Min pt to escape traker

        UseFixedRegion = cms.bool(False), ## Use a fixed region size

        PhiR_UpperLimit_Par1 = cms.double(0.6),
        EtaR_UpperLimit_Par2 = cms.double(0.15),
        Phi_min = cms.double(0.1), ## Lower limit to the region phi

        UseVertex = cms.bool(False), ## use reconstructed vertex instead of beamspot
        MeasurementTrackerName = cms.string("")
    )
)


