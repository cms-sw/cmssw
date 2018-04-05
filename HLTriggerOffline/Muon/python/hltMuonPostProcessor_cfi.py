import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hltMuonPostProcessor  = DQMEDHarvester("DQMGenericClient",
    subDirs           = cms.untracked.vstring('HLT/Muon/Distributions/*'),
    verbose           = cms.untracked.uint32(0),
    outputFileName    = cms.untracked.string(''),
    resolution        = cms.vstring(),                                    
    efficiency        = cms.vstring(),
    efficiencyProfile = cms.untracked.vstring(),
)

