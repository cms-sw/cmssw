import FWCore.ParameterSet.Config as cms

from ..modules.pfTICL_cfi import *

ticlPFSequence = cms.Sequence(pfTICL)
