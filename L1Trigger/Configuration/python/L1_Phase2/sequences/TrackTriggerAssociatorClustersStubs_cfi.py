import FWCore.ParameterSet.Config as cms

from ..modules.TTClusterAssociatorFromPixelDigis_cfi import *
from ..modules.TTStubAssociatorFromPixelDigis_cfi import *

TrackTriggerAssociatorClustersStubs = cms.Sequence(TTClusterAssociatorFromPixelDigis+TTStubAssociatorFromPixelDigis)
