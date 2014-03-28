import FWCore.ParameterSet.Config as cms


defaultTrackSelectionBlock = cms.PSet(
    trackSelection = cms.PSet(
        totalHitsMin = cms.uint32(8),
        maxPixelBarrelLayer = cms.uint32(4),
        maxPixelEndcapLayer = cms.uint32(3),
        jetDeltaRMax = cms.double(0.3),
        qualityClass = cms.string('highPurity'),
        pixelHitsMin = cms.uint32(2),
        maxDistToAxis = cms.double(0.2),
        maxDecayLen = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip2dValMax = cms.double(99999.9),
        ptMin = cms.double(1.0),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        sip2dValMin = cms.double(-99999.9),
        normChi2Max = cms.double(99999.9),
        useVariableJTA = cms.bool(False) 
        )
    )
