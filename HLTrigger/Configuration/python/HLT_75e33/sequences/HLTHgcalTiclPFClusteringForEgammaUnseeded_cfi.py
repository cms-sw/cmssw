import FWCore.ParameterSet.Config as cms

from ..tasks.HLTHgcalTiclPFClusteringForEgammaUnseededTask_cfi import *

HLTHgcalTiclPFClusteringForEgammaUnseeded = cms.Sequence(HLTHgcalTiclPFClusteringForEgammaUnseededTask)
