import FWCore.ParameterSet.Config as cms

MaxCCCLostHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxCCCLostHitsTrajectoryFilter'),
    maxCCCLostHits = cms.int32(3),
    minGoodStripCharge = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutLoose')
    )
)