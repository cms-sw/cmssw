import FWCore.ParameterSet.Config as cms

from DQM.GEM.GEMDQMHarvester_cfi import *
from DQMOffline.Muon.gemEfficiencyHarvester_cfi import *

gemClients = cms.Sequence(
    GEMDQMHarvester *
    gemEfficiencyHarvesterTightGlb *
    gemEfficiencyHarvesterSta
    )
