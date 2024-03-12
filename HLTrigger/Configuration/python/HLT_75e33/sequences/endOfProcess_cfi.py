import FWCore.ParameterSet.Config as cms

from ..modules.MEtoEDMConverter_cfi import *

endOfProcess = cms.Sequence(MEtoEDMConverter)
