import FWCore.ParameterSet.Config as cms

from ..modules.DisplacedVertexProducer_cfi import *
from ..modules.TTClusterAssociatorFromPixelDigis_cfi import *
from ..modules.TTStubAssociatorFromPixelDigis_cfi import *
from ..modules.TTTrackAssociatorFromPixelDigis_cfi import *
from ..modules.TTTrackAssociatorFromPixelDigisExtended_cfi import *

L1TrackTriggerMCTruthTask = cms.Task(DisplacedVertexProducer, TTClusterAssociatorFromPixelDigis, TTStubAssociatorFromPixelDigis, TTTrackAssociatorFromPixelDigis, TTTrackAssociatorFromPixelDigisExtended)
