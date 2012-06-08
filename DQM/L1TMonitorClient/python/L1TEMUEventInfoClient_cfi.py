# L1 Emulator Event Info client cfi
#
#   authors previous versions - see CVS
#
#   V.M. Ghete 2010-10-22 revised version of L1 emulator DQM



import FWCore.ParameterSet.Config as cms

l1temuEventInfoClient = cms.EDAnalyzer("L1TEMUEventInfoClient",
    monitorDir = cms.untracked.string(''),
    
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
    #     - SystemMask:   system mask: if 1, all quality tests for the system 
    #                     are masked in the summary plot
    #     - SystemFolder: the folder where the ErrorFlag histogram is looked for
    #
    # the position in the parameter set gives, in reverse order, the position in the reportSummaryMap
    # in the emulator column (left column)
    L1Systems = cms.VPSet(
                    cms.PSet(
                        SystemLabel = cms.string("ECAL_TPG"),
                        HwValLabel = cms.string("ETP"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string(""),
                        QualityTestName = cms.vstring("L1EmulatorECALErrorFlagQT"),
                        QualityTestHist = cms.vstring("ETPErrorFlag"),
                        QualityTestSummaryEnabled = cms.vuint32(0)                       
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("HCAL_TPG"),
                        HwValLabel = cms.string("HTP"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string(""),
                        QualityTestName = cms.vstring("L1EmulatorHCALErrorFlagQT"),
                        QualityTestHist = cms.vstring("HTPErrorFlag"),
                        QualityTestSummaryEnabled = cms.vuint32(0)                       
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("RCT"),
                        HwValLabel = cms.string("RCT"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string(""),
                        QualityTestName = cms.vstring("L1EmulatorRCTErrorFlagQT"),
                        QualityTestHist = cms.vstring("RCTErrorFlag"),
                        QualityTestSummaryEnabled = cms.vuint32(1)                       
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("GCT"),
                        HwValLabel = cms.string("GCT"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string(""),
                        QualityTestName = cms.vstring("L1EmulatorGCTErrorFlagQT"),
                        QualityTestHist = cms.vstring("GCTErrorFlag"),
                        QualityTestSummaryEnabled = cms.vuint32(1)                       
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("DT_TPG"),
                        HwValLabel = cms.string("DTP"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string(""),
                        QualityTestName = cms.vstring("L1EmulatorDTTPGErrorFlagQT"),
                        QualityTestHist = cms.vstring("DTPErrorFlag"),
                        QualityTestSummaryEnabled = cms.vuint32(0)                       
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("DTTF"),
                        HwValLabel = cms.string("DTF"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string(""),
                        QualityTestName = cms.vstring("L1EmulatorDTTFErrorFlagQT"),
                        QualityTestHist = cms.vstring("DTFErrorFlag"),
                        QualityTestSummaryEnabled = cms.vuint32(1)                       
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("CSC_TPG"),
                        HwValLabel = cms.string("CTP"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string(""),
                        QualityTestName = cms.vstring("L1EmulatorCSCTPGErrorFlagQT"),
                        QualityTestHist = cms.vstring("CTPErrorFlag"),
                        QualityTestSummaryEnabled = cms.vuint32(0)                       
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("CSCTF"),
                        HwValLabel = cms.string("CTF"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string("L1TEMU/CSCTFexpert"),
                        QualityTestName = cms.vstring("L1EMulatorCSCTFDPhi12_ptLut",
                                                      "L1EMulatorCSCTFDPhi23_ptLut",
                                                      "L1EMulatorCSCTFEta_ptLut",
                                                      "L1EMulatorCSCTFMode_ptLut",
                                                      "L1EMulatorCSCTFFRBit_ptLut",

                                                      "L1EMulatorCSCTFPhi",
                                                      "L1EMulatorCSCTFEta",
                                                      "L1EMulatorCSCTFOcc",
                                                      "L1EMulatorCSCTFPt",
                                                      "L1EMulatorCSCTFQual",

                                                      "L1EMulatorCSCTF_dtStubPhi"),
                        QualityTestHist = cms.vstring("pt1Comp_1d",
                                                      "pt2Comp_1d",
                                                      "pt3Comp_1d",
                                                      "pt4Comp_1d",
                                                      "pt5Comp_1d",

                                                      "phiComp_1d",
                                                      "etaComp_1d",
                                                      "occComp_1d",
                                                      "ptComp_1d",
                                                      "qualComp_1d",

                                                      "dtStubPhi_1d"),
                        QualityTestSummaryEnabled = cms.vuint32(1,1,1,1,1,
                                                                1,1,1,1,1,
                                                                1)         
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("RPC"),
                        HwValLabel = cms.string("RPC"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string(""),
                        QualityTestName = cms.vstring("L1EmulatorRPCErrorFlagQT"),
                        QualityTestHist = cms.vstring("RPCErrorFlag"),
                        QualityTestSummaryEnabled = cms.vuint32(1)                       
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("GMT"),
                        HwValLabel = cms.string("GMT"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string(""),
                        QualityTestName = cms.vstring("L1EmulatorGMTErrorFlagQT"),
                        QualityTestHist = cms.vstring("GMTErrorFlag"),
                        QualityTestSummaryEnabled = cms.vuint32(1)                       
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("GT"),
                        HwValLabel = cms.string("GT"),
                        SystemMask  = cms.uint32(0),
                        SystemFolder = cms.string("L1TEMU/GTexpert"),
                        QualityTestName = cms.vstring("L1EmulatorGTErrorFlagQT", "L1EmulatorDaqFdlDataEmulQT"),
                        QualityTestHist = cms.vstring("GTErrorFlag", "Daq_FdlDataEmul_0"),
                        QualityTestSummaryEnabled = cms.vuint32(1, 0)                       
                        )
                        
                     ),
    #
    # for each L1 trigger object, give:
    #     - ObjectLabel:  object label as used in enum L1GtObject
    #     - ObjectMask: emulator mask: if 1, the system is masked in the summary plot
    #
    # the position in the parameter set gives, in reverse order, the position in the reportSummaryMap
    # in the trigger object column (right column)
    L1Objects = cms.VPSet(
                    cms.PSet(
                        ObjectLabel = cms.string("TechTrig"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("GtExternal"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()                       
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("HfRingEtSums"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("HfBitCounts"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("HTM"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("HTT"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("ETM"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("ETT"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("TauJet"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("ForJet"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("CenJet"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("IsoEG"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("NoIsoEG"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string(""),
                        QualityTestName = cms.vstring(),
                        QualityTestHist = cms.vstring(),
                        QualityTestSummaryEnabled = cms.vuint32()
                        ),
                    cms.PSet(
                        ObjectLabel = cms.string("Mu"),
                        ObjectMask  = cms.uint32(0),
                        ObjectFolder = cms.string("L1TEMU/GTexpert"),
                        QualityTestName = cms.vstring("L1EmulatorAlgorithmsMuQT"),
                        QualityTestHist = cms.vstring("GTErrorFlag"),
                        QualityTestSummaryEnabled = cms.vuint32(0)
                        )
                        
                     ),
    #
    # fast over-mask a system: if the name of the system is in the list, the system will be masked
    # (the default mask value is given in L1Systems VPSet)             
    #
    MaskL1Systems = cms.vstring(),
    #
    # fast over-mask an object: if the name of the object is in the list, the object will be masked
    # (the default mask value is given in L1Objects VPSet)             
    # 
    MaskL1Objects =  cms.vstring()   

)
