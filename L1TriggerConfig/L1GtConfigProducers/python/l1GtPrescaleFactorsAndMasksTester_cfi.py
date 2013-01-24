import FWCore.ParameterSet.Config as cms

l1GtPrescaleFactorsAndMasksTester = cms.EDAnalyzer("L1GtPrescaleFactorsAndMasksTester",
                                                   TesterPrescaleFactors=cms.bool(True),
                                                   TesterTriggerMask=cms.bool(True),
                                                   TesterTriggerVetoMask=cms.bool(True),
                                                   RetrieveInBeginRun=cms.bool(True),
                                                   RetrieveInBeginLuminosityBlock=cms.bool(False),
                                                   RetrieveInAnalyze=cms.bool(False),
                                                   PrintInBeginRun=cms.bool(True),
                                                   PrintInBeginLuminosityBlock=cms.bool(False),
                                                   PrintInAnalyze=cms.bool(False),
                                                   # print output
                                                   #   0 std::cout
                                                   #   1 LogTrace
                                                   #   2 LogVerbatim
                                                   #   3 LogInfo
                                                   PrintOutput=cms.untracked.int32(3)
)

