import FWCore.ParameterSet.Config as cms

harvestingdatacertification = cms.EDFilter("HarvestingDataCertification",
    Verbosity = cms.untracked.int32(0),
    Name = cms.untracked.string('HarvestingDataCertification')
)


