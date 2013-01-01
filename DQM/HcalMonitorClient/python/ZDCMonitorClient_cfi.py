import FWCore.ParameterSet.Config as cms
from copy import deepcopy

zdcClient = cms.EDAnalyzer("ZDCMonitorClient",
                           debug=cms.untracked.int32(0),
                           inputFile=cms.untracked.string(""),
                           mergeRuns=cms.untracked.bool(False),
                           cloneME=cms.untracked.bool(False),
                           prescaleFactor=cms.untracked.int32(-1),
                           subSystemFolder           = cms.untracked.string('Hcal/'), # change to "ZDC" at some point, when ZDC DQM process is decoupled form HCAL?
                           ZDCFolder                 = cms.untracked.string("ZDCMonitor_Hcal/"), # This is the subfolder with the subSystemFolder where histograms are kept
                           enableCleanup             = cms.untracked.bool(False),
                           online                    = cms.untracked.bool(False),  # Use to toggle between online/offline-specific functions

                           
                           # These should not be needed by ZDC Monitor Client, at least in the near future -- used for outputting channel status database information for HCAL. 
                           baseHtmlDir    = cms.untracked.string(""),
                           htmlUpdateTime = cms.untracked.int32(0),
                           htmlFirstUpdate = cms.untracked.int32(20),
                           databaseDir = cms.untracked.string(""),
                           databaseUpdateTime = cms.untracked.int32(0),
                           databaseFirstUpdate = cms.untracked.int32(10), # database updates have a 10 minute offset, if updatetime>0
                           
                           # Specify whether LS-by-LS certification should be created
                           saveByLumiSection = cms.untracked.bool(False),

                           ZDC_QIValueForGoodLS = cms.untracked.vdouble(0.8, #The ZDC+ must have at least this high a quality index (QI) to be called good for that Lumi Section (LS) 
                                                                        0.8  #The ZDC- must have at least this high a quality index (QI) to be called good for that Lumi Sectoin (LS)
                                                                        ),
                           
                          )
