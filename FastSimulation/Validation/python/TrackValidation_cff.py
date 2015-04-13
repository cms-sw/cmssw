import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.Validation_cff import *


prevalidation = cms.Sequence( 
    simHitTPAssocProducer
    *tracksValidationSelectors
    )

validation = cms.Sequence(
    trackingTruthValid
    +tracksValidationFS
    )

