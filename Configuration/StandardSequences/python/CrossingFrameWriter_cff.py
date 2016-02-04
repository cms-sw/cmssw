import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.cfwriter_cfi import *

pcfw = cms.Sequence(cms.SequencePlaceholder("mix")*cfWriter)
