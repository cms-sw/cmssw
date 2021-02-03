import FWCore.ParameterSet.Config as cms

from ..modules.muonSeededTracksInOutSelector_cfi import *

muonSeededStepExtraInOutTask = cms.Task(muonSeededTracksInOutSelector)
