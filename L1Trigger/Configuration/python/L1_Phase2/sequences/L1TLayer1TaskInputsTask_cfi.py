import FWCore.ParameterSet.Config as cms

from ..modules.l1tPFClustersFromCombinedCaloHCal_cfi import *
from ..modules.l1tPFClustersFromCombinedCaloHF_cfi import *
from ..modules.l1tPFClustersFromL1EGClusters_cfi import *
from ..modules.l1tPFTracksFromL1Tracks_cfi import *
from ..modules.l1tPFTracksFromL1TracksExtended_cfi import *

L1TLayer1TaskInputsTask = cms.Task(l1tPFClustersFromCombinedCaloHCal, l1tPFClustersFromCombinedCaloHF, l1tPFClustersFromL1EGClusters, l1tPFTracksFromL1Tracks, l1tPFTracksFromL1TracksExtended)
