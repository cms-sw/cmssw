import FWCore.ParameterSet.Config as cms

from ..sequences.ecalLocalRecoSequence_cfi import *
from ..sequences.hcalLocalRecoSequence_cfi import *

calolocalreco = cms.Sequence(ecalLocalRecoSequence+hcalLocalRecoSequence)
