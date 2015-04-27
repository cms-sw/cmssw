import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.cfwriter_cfi import *
from SimGeneral.MixingModule.psimVertexFilter_cfi import *

pcfw = cms.Sequence(cms.SequencePlaceholder("mix")*cfWriter)

pcfsw = cms.Sequence(cms.SequencePlaceholder("mix")*psimVertexFilter)
