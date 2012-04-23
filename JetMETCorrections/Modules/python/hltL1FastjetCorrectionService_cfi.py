# HLT template for L1FastjetCorrectionService

import FWCore.ParameterSet.Config as cms

# L1 (FastJet PU Subtraction) correction service
hltL1FastjetCorrectionService = cms.ESSource('L1FastjetCorrectionService',
    appendToDataLabel = cms.string( '' ),
    era       = cms.string( 'Jec10V1' ),
    level     = cms.string( 'L1FastJet' ),
    algorithm = cms.string( 'AK5Calo' ),
    section   = cms.string( '' ),
    srcRho    = cms.InputTag( 'hltKT6CaloJets', 'rho' ),
    useCondDB = cms.untracked.bool( False )
)
