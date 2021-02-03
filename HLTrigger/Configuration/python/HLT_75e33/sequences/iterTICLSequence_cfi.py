import FWCore.ParameterSet.Config as cms

from ..tasks.iterTICLTask_cfi import *

iterTICLSequence = cms.Sequence(iterTICLTask)
