import FWCore.ParameterSet.Config as cms

hcalClient = cms.EDAnalyzer("HcalMonitorClient",
                            debug=cms.untracked.int32(0),
                            inputFile=cms.untracked.string(""),
                            mergeRuns=cms.untracked.bool(False),
                            cloneME=cms.untracked.bool(False),
                            prescaleFactor=cms.untracked.int32(-1),
                            subSystemFolder=cms.untracked.string("Hcal/"),
                            enableCleanup=cms.untracked.bool(False),
                            
                            baseHtmlDir = cms.untracked.string(""),
                            htmlUpdateTime = cms.untracked.int32(0),
                            htmlFirstUpdate = cms.untracked.int32(20),
                            databaseDir = cms.untracked.string(""),
                            databaseUpdateTime = cms.untracked.int32(0),
                            databaseFirstUpdate = cms.untracked.int32(10), # database updates have a 10 minute offset, if updatetime>0

                            # Specify whether LS-by-LS certification should be created
                            saveByLumiSection = cms.untracked.bool(False),

                            online = cms.untracked.bool(False),  # set to true only for online DQM running
                            # When enabled, this checks for NaN values in the channel status, and counts any such channels as errors in the reportSummary:
                            UseBadChannelStatusInSummary = cms.untracked.bool(False),
                            
                            # each client has a 'minerrror' (double) rate
                            # (minimum fraction of events that must be bad to be considered a problem),
                            # a 'minevents' integer
                            # (minimum number of events to be processed before evaluation occurs),
                            # and a BadChannelStatusMask value
                            # (channel status types that should be checked when looking for known bad channels)

                            # if the following are uncommented, they override the defaults on any
                            # clients that are not specified explicitly with their own prefixes
                            # online running -- require only 1 event (offline will require more)
                            minevents            = cms.untracked.int32(250),
                            # minerrorrate         = cms.untracked.double(0.05),
                            Beam_minLS           = cms.untracked.int32(1),

                            # dead cell min events controlled by task in online running
                            DeadCell_minerrorrate = cms.untracked.double(0.05),
                            #DeadCell_minevents    = cms.untracked.int32(10),
                            HotCell_minerrorrate  = cms.untracked.double(0.10),
                            DeadCell_BadChannelStatusMask =  cms.untracked.int32((1<<5) | (1<<1)),
                            RecHit_BadChannelStatusMask =  cms.untracked.int32((1<<5) | (1<<1)),
                            Digi_BadChannelStatusMask  =  cms.untracked.int32((1<<5) | (1<<1)),
                            CoarsePedestal_BadChannelStatusMask = cms.untracked.int32((1<<5) | (1<<6) | (1<<1)),
                            HotCell_BadChannelStatusMask        = cms.untracked.int32((1<<5) | (1<<1)),
                            
                            excludeHOring2_backup          = cms.untracked.bool(True), # This is only a 'backup' result, and is overwritten by what was used by the task when the task information
                                                                                       # can be found in the DQM output. If the task info can't be found, this backup value is used in its place.
                            excludeBadQPLL         = cms.untracked.bool(True),

                            ZDC_QIValueForGoodLS = cms.untracked.vdouble(0.8, #The ZDC+ must have at least this high a quality index (QI) to be called good for that Lumi Section (LS) 
                                                                        0.8  #The ZDC- must have at least this high a quality index (QI) to be called good for that Lumi Sectoin (LS)
                                                                        ),

                            ZDCFolder                 = cms.untracked.string("ZDCMonitor_Hcal/"), # This is the subfolder with the subSystemFolder where histograms are kept

                            # Specify all clients to be run (name = prefix+"Monitor")

                            enabledClients = cms.untracked.vstring(["DeadCellMonitor",
                                                                    "HotCellMonitor",
                                                                    "RecHitMonitor",
                                                                    "DigiMonitor",
                                                                    "RawDataMonitor",
                                                                    "TrigPrimMonitor",
                                                                    "NZSMonitor",
                                                                    "BeamMonitor",
                                                                    "DetDiagPedestalMonitor",
                                                                    "DetDiagLaserMonitor",
                                                                    "DetDiagLEDMonitor",
                                                                    "DetDiagNoiseMonitor",
                                                                    "DetDiagTimingMonitor",
                                                                    "CoarsePedestalMonitor",
                                                                    "Summary"
                                                                    ]
                                                                   ),
                            )
