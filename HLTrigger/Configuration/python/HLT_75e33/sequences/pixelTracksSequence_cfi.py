import FWCore.ParameterSet.Config as cms

from ..tasks.pixelTracksTask_cfi import *

pixelTracksSequence = cms.Sequence(pixelTracksTask)
