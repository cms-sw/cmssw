import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.gemEfficiencyHarvesterCosmics_cfi import *

gemClientsCosmics = cms.Sequence(
    gemEfficiencyHarvesterCosmics *
    gemEfficiencyHarvesterCosmicsOneLeg)
