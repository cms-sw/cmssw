import FWCore.ParameterSet.Config as cms

from DQM.HcalMonitorClient.HcalMonitorClient_cfi import *
from DQM.HcalMonitorClient.ZDCMonitorClient_cfi  import *
hcalOfflineDQMClient = cms.Sequence(hcalClient
                                    + zdcClient  # re-enable once zdc has been tested offline
                                    )

hcalClient.baseHtmlDir       = ''
hcalClient.databaseDir       = ''
hcalClient.minevents         = 500  # Don't count errors when less than 500 events processed
hcalClient.enabledClients    = ["DeadCellMonitor",
                                "HotCellMonitor",
                                "RecHitMonitor",
                                "DigiMonitor",
                                "RawDataMonitor",
                                #"TrigPrimMonitor", # can't enable trig prim monitor, because no trig sim available offline!
                                "NZSMonitor",
                                #"BeamMonitor",  # don't use BeamMonitor, because I don't trust HF lumi error thresholds to remain valid in higher-luminosity runs
                                "DetDiagNoiseMonitor",
                                "Summary"
                                ]
# Enable save-by-lumi-section reportSummaries in offline only for now
hcalClient.saveByLumiSection=True
