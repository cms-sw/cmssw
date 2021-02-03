import FWCore.ParameterSet.Config as cms

from ..modules.glbTrackQual_cfi import *
from ..modules.muonEcalDetIds_cfi import *
from ..modules.muons1stStep_cfi import *
from ..modules.muonShowerInformation_cfi import *

muonIdProducerTask = cms.Task(glbTrackQual, muonEcalDetIds, muonShowerInformation, muons1stStep)
