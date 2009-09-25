import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixHiSignal_cff import *

phisignal = cms.Sequence(hiSignalSequence)

from SimGeneral.TrackingAnalysis.HiTrackingParticles_cff import *
