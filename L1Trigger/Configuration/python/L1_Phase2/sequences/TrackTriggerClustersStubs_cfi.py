import FWCore.ParameterSet.Config as cms

from ..modules.TTClustersFromPhase2TrackerDigis_cfi import *
from ..modules.TTStubsFromPhase2TrackerDigis_cfi import *

TrackTriggerClustersStubs = cms.Sequence(TTClustersFromPhase2TrackerDigis+TTStubsFromPhase2TrackerDigis)
