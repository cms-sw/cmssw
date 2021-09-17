import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.gemEfficiencyHarvester_cfi import *

gemClients = cms.Sequence(
    gemEfficiencyHarvesterTightGlb *
    gemEfficiencyHarvesterSta)
