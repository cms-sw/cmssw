import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dtTnPEfficiencyClient = DQMEDHarvester("TnPEfficiencyClient",
                                       #Histogram names listed as "passProbeHistoName:failProbeHistoName"
                                       subsystem = cms.untracked.string("DT"),
                                       histoNames = cms.untracked.vstring("DT_nPassingProbePerCh_W-2:DT_nFailingProbePerCh_W-2",
                                                                          "DT_nPassingProbePerCh_W-1:DT_nFailingProbePerCh_W-1",
                                                                          "DT_nPassingProbePerCh_W0:DT_nFailingProbePerCh_W0",
                                                                          "DT_nPassingProbePerCh_W1:DT_nFailingProbePerCh_W1",
                                                                          "DT_nPassingProbePerCh_W2:DT_nFailingProbePerCh_W2",
                                                                          "DT_nPassingProbe_allCh:DT_nFailingProbe_allCh"),
                                       diagnosticPrescale = cms.untracked.int32(1))
