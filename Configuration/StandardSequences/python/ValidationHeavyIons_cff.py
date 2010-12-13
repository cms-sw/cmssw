import FWCore.ParameterSet.Config as cms

from Validation.Configuration.ValidationHI_cff import *

prevalidation = cms.Sequence(cms.SequencePlaceholder("mix"))
validation = cms.Sequence(validationHI)

