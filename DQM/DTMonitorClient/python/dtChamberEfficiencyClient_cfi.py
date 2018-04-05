import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dtChamberEfficiencyClient = DQMEDHarvester("DTChamberEfficiencyClient",
                                           diagnosticPrescale = cms.untracked.int32(1))
