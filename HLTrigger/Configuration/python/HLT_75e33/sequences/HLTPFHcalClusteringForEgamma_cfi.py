import FWCore.ParameterSet.Config as cms

from ..tasks.HLTPFHcalClusteringForEgammaTask_cfi import *

HLTPFHcalClusteringForEgamma = cms.Sequence(
    HLTPFHcalClusteringForEgammaTask
)
