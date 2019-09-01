import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

triggerMatchEffPlotterTightMiniAOD = DQMEDHarvester("TriggerMatchEfficiencyPlotter",
                                          folder = cms.string("Muons_miniAOD/TriggerMatchMonitor"),
                                          triggerhistName1 = cms.string('IsoMu24'),
                                          triggerhistName2 = cms.string('Mu50')
                                          )


