#
# L1 Emulator Event Info client cff
#
#   authors previous versions - see CVS
#
#   V.M. Ghete 2010-10-22 revised version of L1 emulator DQM
#   V.M. Ghete 2012-06-01 get l1temuEventInfoClient by cloning l1tEventInfoClient

import FWCore.ParameterSet.Config as cms

import DQM.L1TMonitorClient.L1TEventInfoClient_cfi
l1temuEventInfoClient = DQM.L1TMonitorClient.L1TEventInfoClient_cfi.l1tEventInfoClient.clone()


l1EmulatorEventInfoClient = cms.Sequence(l1temuEventInfoClient)


# adapt L1TEventInfoClient_cfi to L1TEMU 

l1temuEventInfoClient.monitorDir = cms.untracked.string("L1TEMU")

    # decide when to run and update the results of the quality tests
    # retrieval of quality test results must be consistent with the event / LS / Run execution
    # 
l1temuEventInfoClient.runInEventLoop=cms.untracked.bool(False)
l1temuEventInfoClient.runInEndLumi=cms.untracked.bool(True)
l1temuEventInfoClient.runInEndRun=cms.untracked.bool(True)
l1temuEventInfoClient.runInEndJob=cms.untracked.bool(False)

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
    
l1temuEventInfoClient.L1Systems = cms.VPSet(
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
                                QualityTestName = cms.string("L1EmulatorRCTErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/RCT/RCTErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),   
                            ###                                                   
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTeff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/RegionData/rctRegEff2D"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            #                     
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTeff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/IsoEm/rctIsoEmEff1"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTeff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/NisoEm/rctNisoEmEff1"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            #                     
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTeff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/IsoEm/rctIsoEmEff2"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTeff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/NisoEm/rctNisoEmEff2"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            ###      
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTeff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/RegionData/rctRegEff1D"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            #                     
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTeff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/IsoEm/rctIsoEmEff1oneD"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTeff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/NisoEm/rctNisoEmEff1oneD"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                                 
                            #                     
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTeff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/IsoEm/rctIsoEmEff2oneD"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTeff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/NisoEm/rctNisoEmEff2oneD"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            ###                     
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/RegionData/rctRegIneff2D"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            #                     
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/IsoEm/rctIsoEmIneff"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                  
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/NisoEm/rctNisoEmIneff"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),  
                            #                               
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/IsoEm/rctIsoEmIneff2"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                  
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/NisoEm/rctNisoEmIneff2"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),  
                            #                     
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/RegionData/rctRegOvereff2D"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            #                               
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/IsoEm/rctIsoEmOvereff"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                  
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff2DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/NisoEm/rctNisoEmOvereff"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),  
                            ###                     
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/RegionData/rctRegIneff1D"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/RegionData/rctRegSpIneff1D"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            #                               
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/IsoEm/rctIsoEmIneff1D"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                  
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/NisoEm/rctNisoEmIneff1D"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),  
                            #                               
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/IsoEm/rctIsoEmIneff2oneD"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                  
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/NisoEm/rctNisoEmIneff2oneD"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),  
                            #
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/RegionData/rctRegOvereff1D"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            #                               
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/IsoEm/rctIsoEmOvereff1D"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),                                  
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorRCTineff1DErrorQTest"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeRCT/NisoEm/rctNisoEmOvereff1D"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )                             
                            )
                        ),                                 
                    cms.PSet(
                        SystemLabel = cms.string("GCT"),
                        HwValLabel = cms.string("GCT"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorGCTErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCT/GCTErrorFlag"),
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
                    cms.PSet(
                        SystemLabel = cms.string("DTTF"),
                        HwValLabel = cms.string("DTF"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorDTTFErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/DTTF/DTFErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
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
                                QualityTestName = cms.string("L1EMulatorCSCTFDPhi12_ptLut"),
                                QualityTestHist = cms.string("L1TEMU/CSCTFexpert/pt1Comp_1d"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EMulatorCSCTFDPhi23_ptLut"),
                                QualityTestHist = cms.string("L1TEMU/CSCTFexpert/pt2Comp_1d"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EMulatorCSCTFEta_ptLut"),
                                QualityTestHist = cms.string("L1TEMU/CSCTFexpert/pt3Comp_1d"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EMulatorCSCTFMode_ptLut"),
                                QualityTestHist = cms.string("L1TEMU/CSCTFexpert/pt4Comp_1d"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EMulatorCSCTFFRBit_ptLut"),
                                QualityTestHist = cms.string("L1TEMU/CSCTFexpert/pt5Comp_1d"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EMulatorCSCTFPhi"),
                                QualityTestHist = cms.string("L1TEMU/CSCTFexpert/phiComp_1d"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EMulatorCSCTFEta"),
                                QualityTestHist = cms.string("L1TEMU/CSCTFexpert/etaComp_1d"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EMulatorCSCTFOcc"),
                                QualityTestHist = cms.string("L1TEMU/CSCTFexpert/occComp_1d"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EMulatorCSCTFPt"),
                                QualityTestHist = cms.string("L1TEMU/CSCTFexpert/ptComp_1d"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EMulatorCSCTFQual"),
                                QualityTestHist = cms.string("L1TEMU/CSCTFexpert/qualComp_1d"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EMulatorCSCTF_dtStubPhi"),
                                QualityTestHist = cms.string("L1TEMU/CSCTFexpert/dtStubPhi_1d"),
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
                                QualityTestName = cms.string("L1EmulatorRPCErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/RPC/RPCErrorFlag"),
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
                                QualityTestName = cms.string("L1EmulatorGMTErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GMT/GMTErrorFlag"),
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
                                QualityTestName = cms.string("L1EmulatorGTErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GTexpert/GTErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        )
                    )                                 

    #
    # for each L1 trigger object, give:
    #     - ObjectLabel:  object label as used in enum L1GtObject
    #     - ObjectDisable: emulator mask: if 1, the system is masked in the summary plot
    #
    # the position in the parameter set gives, in reverse order, the position in the reportSummaryMap
    # in the trigger object column (right column)
l1temuEventInfoClient.L1Objects = cms.VPSet(
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
                                QualityTestName = cms.string("L1EmulatorHfRingEtSumsGctErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCTexpert/HFSums/HFSumsErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("HfBitCounts"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorHfBitCountsGctErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCTexpert/HFCnts/HFCntsErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("HTM"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorHtmGctErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCTexpert/MHT/MHTErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("HTT"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorHttGctErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCTexpert/HT/HTErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("ETM"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorEtmGctErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCTexpert/MET/METErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("ETT"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorEttGctErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCTexpert/ET/ETErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("TauJet"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorTauJetGctErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCTexpert/TauJet/TauJetErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("ForJet"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorForJetGctErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCTexpert/ForJet/ForJetErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("CenJet"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorCenJetGctErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCTexpert/CenJet/CenJetErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("IsoEG"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorIsoEGGctErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCTexpert/IsoEM/IsoEMErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("NoIsoEG"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorNoIsoEGGctErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GCTexpert/NoisoEM/NoisoEMErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ),                                 
                    cms.PSet(
                        ObjectLabel = cms.string("Mu"),
                        ObjectDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorMuGmtErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/GMT/GMTErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorMuDttfErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/DTTF/DTFErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("L1EmulatorMuRpcErrorFlagQT"),
                                QualityTestHist = cms.string("L1TEMU/RPC/RPCErrorFlag"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                )
                            )
                        ) 
                    )                                
    #
    # fast over-mask a system: if the name of the system is in the list, the system will be masked
    # (the default mask value is given in L1Systems VPSet)             
    #
l1temuEventInfoClient.DisableL1Systems = cms.vstring()
    #
    # fast over-mask an object: if the name of the object is in the list, the object will be masked
    # (the default mask value is given in L1Objects VPSet)             
    # 
l1temuEventInfoClient.DisableL1Objects =  cms.vstring()   




