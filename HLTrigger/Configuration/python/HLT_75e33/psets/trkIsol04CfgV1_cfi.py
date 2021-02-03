import FWCore.ParameterSet.Config as cms

trkIsol04CfgV1 = cms.PSet(
    barrelCuts = cms.PSet(
        algosToReject = cms.vstring('jetCoreRegionalStep'),
        allowedQualities = cms.vstring(),
        maxDPtPt = cms.double(-1),
        maxDR = cms.double(0.4),
        maxDZ = cms.double(0.2),
        minDEta = cms.double(0.015),
        minDR = cms.double(0.0),
        minHits = cms.int32(-1),
        minPixelHits = cms.int32(-1),
        minPt = cms.double(0.7)
    ),
    endcapCuts = cms.PSet(
        algosToReject = cms.vstring('jetCoreRegionalStep'),
        allowedQualities = cms.vstring(),
        maxDPtPt = cms.double(-1),
        maxDR = cms.double(0.4),
        maxDZ = cms.double(0.2),
        minDEta = cms.double(0.015),
        minDR = cms.double(0.0),
        minHits = cms.int32(-1),
        minPixelHits = cms.int32(-1),
        minPt = cms.double(0.7)
    )
)