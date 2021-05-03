import FWCore.ParameterSet.Config as cms

from ..modules.bcToEFilter_cfi import *
from ..modules.emEnrichingFilter_cfi import *

Gen_QCDEmEnrichingNoBCToEFilter = cms.Path(~bcToEFilter+emEnrichingFilter)
