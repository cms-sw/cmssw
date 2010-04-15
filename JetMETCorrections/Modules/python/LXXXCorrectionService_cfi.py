import FWCore.ParameterSet.Config as cms
ak5CaloL2Relative = cms.ESSource(
    'LXXXCorrectionService',
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK5Calo'),
    section   = cms.string('')
    )

