import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixHiSignalExtended_cff import *

phisignal = cms.Sequence(hiSignalSequence)

from SimGeneral.TrackingAnalysis.HiTrackingParticles_cff import *
