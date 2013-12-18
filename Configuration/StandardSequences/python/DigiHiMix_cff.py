import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Digi_cff import *
from SimGeneral.TrackingAnalysis.HiTrackingParticles_cff import *

pdigi = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")*cms.SequencePlaceholder("mix")*doAllDigi*trackingParticles*addPileupInfo)
pdigi_valid = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")*cms.SequencePlaceholder("mix")*doAllDigi*addPileupInfo)





