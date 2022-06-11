import FWCore.ParameterSet.Config as cms

from DQM.GEM.GEMDQMHarvester_cfi import *
from DQM.GEM.gemEfficiencyHarvesterCosmics_cff import *

gemClientsCosmics = cms.Sequence(
    GEMDQMHarvester *
    gemEfficiencyHarvesterCosmicsGlb *
    gemEfficiencyHarvesterCosmicsSta
)
