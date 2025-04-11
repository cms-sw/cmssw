import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

L1ScoutingMonitor = DQMEDAnalyzer('L1ScoutingMonitor',
    muonsTag = cms.InputTag('l1ScGmtUnpacker', 'Muon', 'SCHLP'),
    jetsTag = cms.InputTag("l1ScCaloUnpacker", "Jet", "SCHLP"),
    eGammasTag = cms.InputTag("l1ScCaloUnpacker", "EGamma", "SCHLP"),
    tausTag = cms.InputTag("l1ScCaloUnpacker", "Tau", "SCHLP"),
    dqmPath = cms.untracked.string("/L1Scouting/BX/Occupancy")
)
