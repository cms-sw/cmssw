import FWCore.ParameterSet.Config as cms

trkIsol03CfgV1= cms.PSet(
    barrelCuts=cms.PSet(
        minPt=cms.double(0.7),
        maxDR=cms.double(0.3),
        minDR=cms.double(0.00),
        minDEta=cms.double(0.015),
        maxDZ=cms.double(0.2),
        maxDPtPt=cms.double(-1),
        minHits=cms.int32(-1),
        minPixelHits=cms.int32(-1),
        allowedQualities=cms.vstring(),
        algosToReject=cms.vstring("jetCoreRegionalStep")
        ),
    endcapCuts=cms.PSet(
        minPt=cms.double(0.7),
        maxDR=cms.double(0.3),
        minDR=cms.double(0.00),
        minDEta=cms.double(0.015),
        maxDZ=cms.double(0.2),
        maxDPtPt=cms.double(-1),
        minHits=cms.int32(-1),
        minPixelHits=cms.int32(-1),
        allowedQualities=cms.vstring(),
        algosToReject=cms.vstring("jetCoreRegionalStep")
        )
    )
trkIsol04CfgV1 = trkIsol03CfgV1.clone()
trkIsol04CfgV1.barrelCuts.maxDR = cms.double(0.4)
trkIsol04CfgV1.endcapCuts.maxDR = cms.double(0.4)

trkIsol03CfgV2= cms.PSet(
    barrelCuts=cms.PSet(
        minPt=cms.double(1.0),
        maxDR=cms.double(0.3),
        minDR=cms.double(0.0),
        minDEta=cms.double(0.005),
        maxDZ=cms.double(0.1),
        maxDPtPt=cms.double(0.1),
        minHits=cms.int32(8),
        minPixelHits=cms.int32(1),
        allowedQualities=cms.vstring(),
        algosToReject=cms.vstring()
        ),
    endcapCuts=cms.PSet(
        minPt=cms.double(1.0),
        maxDR=cms.double(0.3),
        minDR=cms.double(0.0),
        minDEta=cms.double(0.005),
        maxDZ=cms.double(0.5),
        maxDPtPt=cms.double(0.1),
        minHits=cms.int32(8),
        minPixelHits=cms.int32(1),
        allowedQualities=cms.vstring(),
        algosToReject=cms.vstring()
        )
    )
trkIsol04CfgV2 = trkIsol03CfgV2.clone()
trkIsol04CfgV2.barrelCuts.maxDR = cms.double(0.4)
trkIsol04CfgV2.endcapCuts.maxDR = cms.double(0.4)

trkIsol03CfgV3= cms.PSet(
    barrelCuts=cms.PSet(
        minPt=cms.double(1.0),
        maxDR=cms.double(0.3),
        minDR=cms.double(0.0),
        minDEta=cms.double(0.005),
        maxDZ=cms.double(0.1),
        maxDPtPt=cms.double(0.1),
        minHits=cms.int32(8),
        minPixelHits=cms.int32(1),
        allowedQualities=cms.vstring(),
        algosToReject=cms.vstring("jetCoreRegionalStep")
        ),
    endcapCuts=cms.PSet(
        minPt=cms.double(1.0),
        maxDR=cms.double(0.3),
        minDR=cms.double(0.0),
        minDEta=cms.double(0.005),
        maxDZ=cms.double(0.5),
        maxDPtPt=cms.double(0.1),
        minHits=cms.int32(8),
        minPixelHits=cms.int32(1),
        allowedQualities=cms.vstring(),
        algosToReject=cms.vstring("jetCoreRegionalStep")
        )
    )
trkIsol04CfgV3 = trkIsol03CfgV3.clone()
trkIsol04CfgV3.barrelCuts.maxDR = cms.double(0.4)
trkIsol04CfgV3.endcapCuts.maxDR = cms.double(0.4)

