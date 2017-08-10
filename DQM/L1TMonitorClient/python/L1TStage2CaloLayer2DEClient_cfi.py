import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tStage2CaloLayer2DEClient = DQMEDHarvester("L1TStage2CaloLayer2DEClient",
                  monitorDir = cms.untracked.string('L1TEMU/L1TStage2CaloLayer2DERatio'),
                  inputDataDir = cms.untracked.string('L1T/L1TStage2CaloLayer2'),
                  inputEmulDir = cms.untracked.string('L1TEMU/L1TStage2CaloLayer2EMU')
)


