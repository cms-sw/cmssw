# HLT template for L1FastjetCorrectionESProducer

import FWCore.ParameterSet.Config as cms

# L1 (FastJet PU Subtraction) correction
hltL1FastjetCorrectionESProducer = cms.ESProducer('L1FastjetCorrectionESProducer',
    appendToDataLabel = cms.string( '' ),
    level     = cms.string( 'L1FastJet' ),
    algorithm = cms.string( 'AK4Calo' ),
    srcRho    = cms.InputTag( 'fixedGridRhoFastjetAllCalo' )
)
