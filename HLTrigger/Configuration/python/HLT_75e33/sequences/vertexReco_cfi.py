import FWCore.ParameterSet.Config as cms

from ..tasks.vertexRecoTask_cfi import *

vertexReco = cms.Sequence(vertexRecoTask)
