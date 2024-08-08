import FWCore.ParameterSet.Config as cms

from ..modules.MEtoEDMConverter_cfi import *

HLTEndOfProcess = cms.Sequence(MEtoEDMConverter)
