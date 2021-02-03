import FWCore.ParameterSet.Config as cms

from ..tasks.muonGlobalRecoTask_cfi import *

muonGlobalReco = cms.Sequence(muonGlobalRecoTask)
