import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tStage2CaloLayer2DEClientSummary = DQMEDHarvester("L1TStage2CaloLayer2DEClientSummary",
                  monitorDir = cms.untracked.string('L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2')
                  )


