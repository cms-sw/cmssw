import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dataIntegrityTest = DQMEDHarvester("DTDataIntegrityTest",
                                   diagnosticPrescale = cms.untracked.int32(1)
)


