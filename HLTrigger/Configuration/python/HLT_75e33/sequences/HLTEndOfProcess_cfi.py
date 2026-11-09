import FWCore.ParameterSet.Config as cms

from ..modules.hltMEtoEDMConverter_cfi import *

HLTEndOfProcess = cms.Sequence(hltMEtoEDMConverter)
