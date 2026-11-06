import FWCore.ParameterSet.Config as cms

from ..modules.l1tLayer1_cfi import *
from ..modules.l1tLayer1Barrel_cfi import *
from ..modules.l1tLayer1BarrelExtended_cfi import *
from ..modules.l1tLayer1EG_cfi import *
from ..modules.l1tLayer1EGElliptic_cfi import *
from ..modules.l1tLayer1Extended_cfi import *
from ..modules.l1tLayer1HF_cfi import *
from ..modules.l1tLayer1HGCal_cfi import *
from ..modules.l1tLayer1HGCalElliptic_cfi import *
from ..modules.l1tLayer1HGCalExtended_cfi import *
from ..modules.l1tLayer1HGCalNoTK_cfi import *

L1TLayer1Task = cms.Task(l1tLayer1, l1tLayer1Barrel, l1tLayer1BarrelExtended, l1tLayer1EG, l1tLayer1EGElliptic, l1tLayer1Extended, l1tLayer1HF, l1tLayer1HGCal, l1tLayer1HGCalElliptic, l1tLayer1HGCalExtended, l1tLayer1HGCalNoTK)
