# HLT template for LXXXCorrectionService

import FWCore.ParameterSet.Config as cms

# template for L2 (relative eta-conformity) and L3 (absolute) correction services
hltLXXXCorrectionService = cms.ESSource( 'LXXXCorrectionService',
    appendToDataLabel = cms.string( '' ),
    era       = cms.string( '' ),
    level     = cms.string( '' ),           # "L2Relative" or "L3Absolute"
    algorithm = cms.string( 'AK5Calo' ),
    section   = cms.string( '' ),
    useCondDB = cms.untracked.bool( True )
)
