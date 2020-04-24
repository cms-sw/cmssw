# HLT template for LXXXCorrectionESProducer

import FWCore.ParameterSet.Config as cms

# template for L2 (relative eta-conformity) and L3 (absolute) correction
hltLXXXCorrectionESProducer = cms.ESProducer( 'LXXXCorrectionESProducer',
    appendToDataLabel = cms.string( '' ),
    level     = cms.string( '' ),           # "L2Relative" or "L3Absolute"
    algorithm = cms.string( 'AK4Calo' )
)
