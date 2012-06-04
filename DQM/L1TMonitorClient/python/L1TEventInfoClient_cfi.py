# L1 Trigger Event Info client cfi
# 
#   The cfi can be used, with appropriate settings, for both L1T and L1TEMU. 
#   Default version in cfi: L1T event client
#
#   authors previous versions - see CVS
#
#   V.M. Ghete 2010-10-22 revised version of L1T DQM and L1TEMU DQM



import FWCore.ParameterSet.Config as cms

l1tEventInfoClient = cms.EDAnalyzer("L1TEventInfoClient",
    monitorDir = cms.untracked.string("L1T"),
    
    # decide when to run and update the results of the quality tests
    # retrieval of quality test results must be consistent with the event / LS / Run execution
    # 
    runInEventLoop=cms.untracked.bool(False),
    runInEndLumi=cms.untracked.bool(True),
    runInEndRun=cms.untracked.bool(True),
    runInEndJob=cms.untracked.bool(False),

    #
    # for each L1 system, give:
    #     - SystemLabel:  system label
    #     - HwValLabel:   system label as used in hardware validation package 
    #                     (the package producing the ErrorFlag histogram)
    #     - SystemDisable:   system disabled: if 1, all quality tests for the system 
    #                     are disabled in the summary plot
    #     - for each quality test:
    #         - QualityTestName: name of quality test
    #         - QualityTestHist: histogram (full path)
    #         - QualityTestSummaryEnabled: 0 if disabled, 1 if enabled in summary plot 
    #
    # the position in the parameter set gives, in reverse order, the position in the reportSummaryMap
    # in the emulator column (left column)
    L1Systems = cms.VPSet(
                    cms.PSet(
                        SystemLabel = cms.string("ECAL_TPG"),
                        HwValLabel = cms.string("ETP"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        SystemLabel = cms.string("HCAL_TPG"),
                        HwValLabel = cms.string("HTP"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        SystemLabel = cms.string("RCT"),
                        HwValLabel = cms.string("RCT"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("HotChannels_RCT_2D"),
                                QualityTestHist = cms.string("L1T/L1TRCT/RctEmIsoEmEtEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                                     
                            cms.PSet(
                                QualityTestName = cms.string("DeadChannels_RCT_2D_loose"),
                                QualityTestHist = cms.string("L1T/L1TRCT/RctEmIsoEmEtEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            #                     
                            cms.PSet(
                                QualityTestName = cms.string("HotChannels_RCT_2D"),
                                QualityTestHist = cms.string("L1T/L1TRCT/RctEmNonIsoEmEtEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                                     
                            cms.PSet(
                                QualityTestName = cms.string("DeadChannels_RCT_2D_loose"),
                                QualityTestHist = cms.string("L1T/L1TRCT/RctEmNonIsoEmEtEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            #                                         
                            cms.PSet(
                                QualityTestName = cms.string("HotChannels_RCT_2D"),
                                QualityTestHist = cms.string("L1T/L1TRCT/RctRegionsEtEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                                     
                            cms.PSet(
                                QualityTestName = cms.string("DeadChannels_RCT_2D_tight"),
                                QualityTestHist = cms.string("L1T/L1TRCT/RctRegionsEtEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            )
                        ),                                 
                    cms.PSet(
                        SystemLabel = cms.string("GCT"),
                        HwValLabel = cms.string("GCT"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("HotChannels_GCT_2D"),
                                QualityTestHist = cms.string("L1T/L1TGCT/IsoEmRankEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                                    
                            cms.PSet(
                                QualityTestName = cms.string("DeadChannels_GCT_2D_loose"),
                                QualityTestHist = cms.string("L1T/L1TGCT/IsoEmRankEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            #                     
                            cms.PSet(
                                QualityTestName = cms.string("HotChannels_GCT_2D"),
                                QualityTestHist = cms.string("L1T/L1TGCT/IsoEmRankEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                                    
                            cms.PSet(
                                QualityTestName = cms.string("DeadChannels_GCT_2D_loose"),
                                QualityTestHist = cms.string("L1T/L1TGCT/NonIsoEmRankEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            #                     
                            cms.PSet(
                                QualityTestName = cms.string("HotChannels_GCT_2D"),
                                QualityTestHist = cms.string("L1T/L1TGCT/AllJetsEtEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                                    
                            cms.PSet(
                                QualityTestName = cms.string("DeadChannels_GCT_2D_tight"),
                                QualityTestHist = cms.string("L1T/L1TGCT/AllJetsEtEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("HotChannels_GCT_2D"),
                                QualityTestHist = cms.string("L1T/L1TGCT/TauJetsEtEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                                    
                            cms.PSet(
                                QualityTestName = cms.string("DeadChannels_GCT_2D_tight"),
                                QualityTestHist = cms.string("L1T/L1TGCT/TauJetsEtEtaPhi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                  
                    cms.PSet(
                        SystemLabel = cms.string("DT_TPG"),
                        HwValLabel = cms.string("DTP"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),     
                    # FIXME  what are the histograms to be tested?                            
                    cms.PSet(
                        SystemLabel = cms.string("DTTF"),
                        HwValLabel = cms.string("DTF"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                  
                    cms.PSet(
                        SystemLabel = cms.string("CSC_TPG"),
                        HwValLabel = cms.string("CTP"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                  
                    cms.PSet(
                        SystemLabel = cms.string("CSCTF"),
                        HwValLabel = cms.string("CTF"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("DeadChannels_CSCTF_2D"),
                                QualityTestHist = cms.string("L1T/L1TCSCTF/CSCTF_Chamber_Occupancies"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("HotChannels_CSCTF_2D"),
                                QualityTestHist = cms.string("L1T/L1TCSCTF/CSCTF_Chamber_Occupancies"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),               
                            cms.PSet(
                                QualityTestName = cms.string("DeadChannels_CSCTF_2D"),
                                QualityTestHist = cms.string("L1T/L1TCSCTF/CSCTF_occupancies"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("HotChannels_CSCTF_2D"),
                                QualityTestHist = cms.string("L1T/L1TCSCTF/CSCTF_occupancies"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )               
                            )
                        ),                                  
                    cms.PSet(
                        SystemLabel = cms.string("RPC"),
                        HwValLabel = cms.string("RPC"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("DeadChannels_RPCTF_2D"),
                                QualityTestHist = cms.string("L1T/L1TRPCTF/RPCTF_muons_eta_phi_bx0"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("HotChannels_RPCTF_2D"),
                                QualityTestHist = cms.string("L1T/L1TRPCTF/RPCTF_muons_eta_phi_bx0"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )               
                            )
                        ),                                  
                    cms.PSet(
                        SystemLabel = cms.string("GMT"),
                        HwValLabel = cms.string("GMT"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("DeadChannels_GMT_2D"),
                                QualityTestHist = cms.string("L1T/L1TGMT/GMT_etaphi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("HotChannels_GMT_2D"),
                                QualityTestHist = cms.string("L1T/L1TGMT/GMT_etaphi"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),               
                            cms.PSet(
                                QualityTestName = cms.string("CompareHist_GMT"),
                                QualityTestHist = cms.string("L1T/L1TGMT/Regional_trigger"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )              
                            )
                        ),                                  
                    cms.PSet(
                        SystemLabel = cms.string("GT"),
                        HwValLabel = cms.string("GT"),
                        SystemDisable  = cms.uint32(0),  
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        )
                    ),                                 

    #
    # for each L1 trigger object, give:
    #     - ObjectLabel:  object label as used in enum L1GtObject
    #     - ObjectDisable: emulator mask: if 1, the system is masked in the summary plot
    #
    # the position in the parameter set gives, in reverse order, the position in the reportSummaryMap
    # in the trigger object column (right column)
    L1Objects = cms.VPSet(
                    cms.PSet(
                        ObjectLabel = cms.string("TechTrig"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("GtExternal"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("HfRingEtSums"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("HfBitCounts"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("HTM"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("HTT"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("ETM"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("ETT"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("TauJet"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("ForJet"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("CenJet"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("IsoEG"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("NoIsoEG"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("Mu"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                )
                            )
                        )
                    ),                                
    #
    # fast over-mask a system: if the name of the system is in the list, the system will be masked
    # (the default mask value is given in L1Systems VPSet)             
    #
    DisableL1Systems = cms.vstring(),
    #
    # fast over-mask an object: if the name of the object is in the list, the object will be masked
    # (the default mask value is given in L1Objects VPSet)             
    # 
    DisableL1Objects =  cms.vstring()   

)
