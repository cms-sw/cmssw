import FWCore.ParameterSet.Config as cms

from ..modules.l1tLayer2Deregionizer_cfi import *
from ..modules.l1tLayer2EG_cfi import *
from ..modules.l1tLayer2EGElliptic_cfi import *

L1TLayer2EGTask = cms.Task(l1tLayer2Deregionizer, l1tLayer2EG, l1tLayer2EGElliptic)
