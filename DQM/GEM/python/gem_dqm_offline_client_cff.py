import FWCore.ParameterSet.Config as cms

from DQM.GEM.GEMDQMHarvester_cfi import *
from DQM.GEM.gemEfficiencyHarvester_cff import *

from DQMOffline.MuonDPG.gemTnPEfficiencyClient_cfi import *

gemClients = cms.Sequence(
    GEMDQMHarvester *
    gemEfficiencyHarvesterTightGlb *
    gemEfficiencyHarvesterSta *
    gemTnPEfficiencyClient
)
