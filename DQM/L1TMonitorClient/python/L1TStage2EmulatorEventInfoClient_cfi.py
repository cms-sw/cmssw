# L1 Emulator Trigger Event Info client cfi
#
#   The cfi can be used, with appropriate settings, for both L1T and L1TEMU.
#   Default version in cfi: L1T event client
#
#   authors previous versions - see CVS
#
#   V.M. Ghete 2010-10-22 revised version of L1T DQM and L1TEMU DQM
#   A.G. Stahl 2017-06-02 first implementation for L1TEMU Stage2 DQM



import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tStage2EmulatorEventInfoClient = DQMEDHarvester("L1TEventInfoClient",
    monitorDir = cms.untracked.string("L1TEMU"),

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
                                ),
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
                                ),
                            )
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("Calo Layer1"),
                        HwValLabel = cms.string("Stage2CaloLayer1"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("Layer1DEMismatchThreshold"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeStage2CaloLayer1/dataEmulSummary"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            )
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("Calo Layer2"),
                        HwValLabel = cms.string("Stage2CaloLayer2"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string(""),
                                QualityTestHist = cms.string(""),
                                QualityTestSummaryEnabled = cms.uint32(0)
                                ),
                            )
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("BMTF"),
                        HwValLabel = cms.string("Stage2BMTF"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("BMTFDE_MismatchRatioMax0p01"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeStage2BMTF/mismatchRatio"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            )
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("OMTF"),
                        HwValLabel = cms.string("Stage2OMTF"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("OMTFDE_MismatchRatioMax0p01"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeStage2OMTF/mismatchRatio"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            )
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("EMTF"),
                        HwValLabel = cms.string("Stage2EMTF"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("EMTFDE_MismatchRatioMax0p01"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeStage2EMTF/mismatchRatio"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            )
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("uGMT"),
                        HwValLabel = cms.string("Stage2uGMT"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("uGMTDE_MismatchRatioMax0"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeStage2uGMT/data_vs_emulator_comparison/mismatchRatio"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("InterMuonsDE_MismatchRatioMax0"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeStage2uGMT/intermediate_muons/BMTF/data_vs_emulator_comparison/mismatchRatio"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("InterMuonsDE_MismatchRatioMax0"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeStage2uGMT/intermediate_muons/OMTF_pos/data_vs_emulator_comparison/mismatchRatio"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("InterMuonsDE_MismatchRatioMax0"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeStage2uGMT/intermediate_muons/OMTF_neg/data_vs_emulator_comparison/mismatchRatio"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("InterMuonsDE_MismatchRatioMax0"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeStage2uGMT/intermediate_muons/EMTF_pos/data_vs_emulator_comparison/mismatchRatio"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            cms.PSet(
                                QualityTestName = cms.string("InterMuonsDE_MismatchRatioMax0"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeStage2uGMT/intermediate_muons/EMTF_neg/data_vs_emulator_comparison/mismatchRatio"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            )
                        ),
                    cms.PSet(
                        SystemLabel = cms.string("uGT"),
                        HwValLabel = cms.string("Stage2uGT"),
                        SystemDisable  = cms.uint32(0),
                        QualityTests = cms.VPSet(
                            cms.PSet(
                                QualityTestName = cms.string("uGTDE_CentralBxMismatchRatio"),
                                QualityTestHist = cms.string("L1TEMU/L1TdeStage2uGT/dataEmulMismatchRatio_CentralBX"),
                                QualityTestSummaryEnabled = cms.uint32(1)
                                ),
                            )
                        ),
                    ),

    #
    # for each L1 trigger object, give:
    #     - ObjectLabel:  object label as used in enum L1GtObject
    #     - ObjectDisable: emulator mask: if 1, the system is masked in the summary plot
    #
    # the position in the parameter set gives, in reverse order, the position in the reportSummaryMap
    # in the trigger object column (right column)
    # N.B. (nick): max(# entries in this VPset, # entries in L1Systems) determines the number of columns drawn
    # hence the ugliness of the overlay if not the same as the overlay code
    # which is here-ish: https://github.com/dmwm/deployment/blob/master/dqmgui/style/L1TStage2RenderPlugin.cc#L143-L199
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
                        ObjectLabel = cms.string("Tau"),
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
