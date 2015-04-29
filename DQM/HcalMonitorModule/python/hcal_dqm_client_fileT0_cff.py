import FWCore.ParameterSet.Config as cms

from DQM.HcalMonitorClient.HcalMonitorClient_cfi import *
hcalOfflineDQMClient = cms.Sequence(hcalClient
                                    # + zdcClient  # re-enable once zdc has been tested offline
                                    )

hcalClient.baseHtmlDir       = ''
hcalClient.databaseDir       = ''
hcalClient.minevents         = 500  # Don't count errors when less than 500 events processed
hcalClient.enabledClients    = ["DeadCellMonitor",
                                "HotCellMonitor",
                                "RecHitMonitor",
                                "DigiMonitor",
                                "RawDataMonitor",
                                "ZDCMonitor",
                                #"TrigPrimMonitor", # can't enable trig prim monitor, because no trig sim available offline!
                                "NZSMonitor",
                                #"BeamMonitor",  # don't use BeamMonitor, because I don't trust HF lumi error thresholds to remain valid in higher-luminosity runs
                                "DetDiagNoiseMonitor",
                                "Summary"
                                ]
#The ZDC+/- must have at least this high a quality index (QI) to be called good for that Lumi Section (LS)
hcalClient.ZDC_QIValueForGoodLS = ZDC_QIValueForGoodLS = cms.untracked.vdouble(0.8, 0.8)
# Enable save-by-lumi-section reportSummaries in offline only for now
hcalClient.saveByLumiSection=True
