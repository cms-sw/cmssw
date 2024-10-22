import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

cscTnPEfficiencyClient = DQMEDHarvester("TnPEfficiencyClient",
                                       #Histogram names listed as "passProbeHistoName:failProbeHistoName"
                                       subsystem = cms.untracked.string("CSC"),
                                       histoNames = cms.untracked.vstring("CSC_nPassingProbe_allCh:CSC_nFailingProbe_allCh",
                                                                          "CSC_nPassingProbe_allCh_1D:CSC_nFailingProbe_allCh_1D",
                                                                          "CSC_nPassingProbe_allCh:CSC_nFailingProbe_allCh"),
                                       diagnosticPrescale = cms.untracked.int32(1))
