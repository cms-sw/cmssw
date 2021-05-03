import FWCore.ParameterSet.Config as cms

from ..tasks.HLTPFClusteringForEgammaUnseededTask_cfi import *

HLTPFClusteringForEgammaUnseeded = cms.Sequence(HLTPFClusteringForEgammaUnseededTask)
