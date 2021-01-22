import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

gemEfficiencyHarvesterTight = DQMEDHarvester('GEMEfficiencyHarvester',
    folder = cms.untracked.string('GEM/GEMEfficiency/TightGlobalMuon'),
    logCategory = cms.untracked.string('GEMEfficiencyHarvesterTight')
)

gemEfficiencyHarvesterSTA = DQMEDHarvester('GEMEfficiencyHarvester',
    folder = cms.untracked.string('GEM/GEMEfficiency/StandaloneMuon'),
    logCategory = cms.untracked.string('GEMEfficiencyHarvesterSTA')
)
