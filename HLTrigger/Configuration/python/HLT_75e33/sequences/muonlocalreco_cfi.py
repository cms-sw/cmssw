import FWCore.ParameterSet.Config as cms

from ..tasks.muonlocalrecoTask_cfi import *

muonlocalreco = cms.Sequence(muonlocalrecoTask)
