import FWCore.ParameterSet.Config as cms

#defaultCuts, set to be V3 barrel, cone of 0.3
_defaultCuts = cms.PSet(
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
)

#V1, used in 90X - 93X
trkIsol03CfgV1 = cms.PSet(
    barrelCuts=_defaultCuts.clone(minPt=0.7,minDEta=0.015,maxDZ=0.2,
                                  maxDPtPt=-1,minHits=-1,minPixelHits=-1),
    endcapCuts=_defaultCuts.clone(minPt=0.7,minDEta=0.015,maxDZ=0.2,
                                  maxDPtPt=-1,minHits=-1,minPixelHits=-1)
    )
trkIsol04CfgV1 = cms.PSet(
    barrelCuts=trkIsol03CfgV1.barrelCuts.clone(maxDR=0.4),
    endcapCuts=trkIsol03CfgV1.endcapCuts.clone(maxDR=0.4)
)

#V2, used by HEEP ID in 2016
trkIsol03CfgV2 = cms.PSet(
    barrelCuts=_defaultCuts.clone(algosToReject = cms.vstring()),
    endcapCuts=_defaultCuts.clone(algosToReject = cms.vstring(),maxDZ=0.5)
)
trkIsol04CfgV2 = cms.PSet(
    barrelCuts=trkIsol03CfgV2.barrelCuts.clone(maxDR=0.4),
    endcapCuts=trkIsol03CfgV2.endcapCuts.clone(maxDR=0.4)
)
