import FWCore.ParameterSet.Config as cms
from copy import deepcopy

zdcClient = cms.EDAnalyzer("ZDCMonitorClient",

                          # Variables for the Overall Client
                          runningStandalone         = cms.untracked.bool(False),
                          Online                    = cms.untracked.bool(False), 
                          databasedir               = cms.untracked.string(''),

                          # maximum number of lumi blocks to appear in some histograms
                          Nlumiblocks = cms.untracked.int32(1000),
                          subSystemFolder           = cms.untracked.string('Hcal/ZDCMonitor'), # change to "ZDC" when code is finalized
                          processName               = cms.untracked.string(''),
                          inputfile                 = cms.untracked.string(''),
                          baseHtmlDir               = cms.untracked.string('.'),
                          MonitorDaemon             = cms.untracked.bool(True),

                          # run actual client either every N events or M lumi blocks (or both)
                          diagnosticPrescaleEvt     = cms.untracked.int32(-1),
                          diagnosticPrescaleLS      = cms.untracked.int32(1),
                          resetFreqEvents           = cms.untracked.int32(-1),
                          resetFreqLS               = cms.untracked.int32(-1),
                          
                          debug                     = cms.untracked.int32(0),
                          showTiming                = cms.untracked.bool(False),
                          )
