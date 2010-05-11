import FWCore.ParameterSet.Config as cms

from Validation.Configuration.ValidationHI_cff import *

validationHeavyIons = cms.Sequence(cms.SequencePlaceholder("mix") + validationHI)

