import FWCore.ParameterSet.Config as cms

hcalZDCMonitorTask = cms.PSet(
        debug                  = cms.untracked.int32(0),
        online                 = cms.untracked.bool(False),
        AllowedCalibTypes      = cms.untracked.vint32(0), # by default, don't include calibration events
        mergeRuns              = cms.untracked.bool(False),
        enableCleanup          = cms.untracked.bool(False),
        subSystemFolder        = cms.untracked.string("Hcal/"),
        TaskFolder             = cms.untracked.string("ZDCMonitor_Hcal/"),
        skipOutOfOrderLS       = cms.untracked.bool(False),
        NLumiBlocks            = cms.untracked.int32(4000),

        # Collections to get

        digiLabel            = cms.InputTag("hcalDigis"),  # what label is used for ZDC?  ZDC digis?
        rechitLabel          = cms.InputTag("zdcreco"),
        makeDiagnostics      = cms.untracked.bool(False),



        ########  JAIMES NEW VARIABLES

        ZDC_OnlineColdThreshold = cms.untracked.int32(250), #can change this later if we wnat each section to have a different number of events before being called cold

        ZDC_OnlineDeadThreshold = cms.untracked.int32(250), #same as above

        ZDC_OfflineColdThreshold = cms.untracked.int32(250), #same as above

        ZDC_OfflineDeadThreshold = cms.untracked.int32(250), #same as above

        ZDC_ColdADCThreshold = cms.untracked.int32(20), #This is the threshold for which a channel will be called cold

        ZDC_ChannelWeighting     = cms.untracked.vdouble(0.1, #ZDC+ EM1 contributes this much to the quality index (QI) for the ZDC+
                                                         0.1, #ZDC+ EM2 contributes this much to the quality index (QI) for the ZDC+
                                                         0.1, #ZDC+ EM3 contributes this much to the quality index (QI) for the ZDC+
                                                         0.1, #ZDC+ EM4 contributes this much to the quality index (QI) for the ZDC+
                                                         0.1, #ZDC+ EM5 contributes this much to the quality index (QI) for the ZDC+
                                                         0.125, #ZDC+ HAD1 contributes this much to the quality index (QI) for the ZDC+
                                                         0.125, #ZDC+ HAD2 contributes this much to the quality index (QI) for the ZDC+
                                                         0.125, #ZDC+ HAD3 contributes this much to the quality index (QI) for the ZDC+
                                                         0.125, #ZDC+ HAD4 contributes this much to the quality index (QI) for the ZDC+
                                                         0.1, #ZDC- EM1 contributes this much to the quality index (QI) for the ZDC-
                                                         0.1, #ZDC- EM2 contributes this much to the quality index (QI) for the ZDC-
                                                         0.1, #ZDC- EM3 contributes this much to the quality index (QI) for the ZDC-
                                                         0.1, #ZDC- EM4 contributes this much to the quality index (QI) for the ZDC-
                                                         0.1, #ZDC- EM5 contributes this much to the quality index (QI) for the ZDC-
                                                         0.125, #ZDC- HAD1 contributes this much to the quality index (QI) for the ZDC-
                                                         0.125, #ZDC- HAD2 contributes this much to the quality index (QI) for the ZDC-
                                                         0.125, #ZDC- HAD3 contributes this much to the quality index (QI) for the ZDC-
                                                         0.125  #ZDC- HAD4 contributes this much to the quality index (QI) for the ZDC-
                                                        ),
        ZDC_AcceptableChannelErrorRates = cms.untracked.vdouble(0.1, #ZDC+ EM1 can have this fractional error rate before being called bad
                                                                0.1, #ZDC+ EM2 can have this fractional error rate before being called bad
                                                                0.1, #ZDC+ EM3 can have this fractional error rate before being called bad
                                                                0.1, #ZDC+ EM4 can have this fractional error rate before being called bad
                                                                0.1, #ZDC+ EM5 can have this fractional error rate before being called bad
                                                                0.1, #ZDC+ HAD1 can have this fractional error rate before being called bad
                                                                0.1, #ZDC+ HAD2 can have this fractional error rate before being called bad
                                                                0.1, #ZDC+ HAD3 can have this fractional error rate before being called bad
                                                                0.1, #ZDC+ HAD4 can have this fractional error rate before being called bad
                                                                0.1, #ZDC- EM1 can have this fractional error rate before being called bad
                                                                0.1, #ZDC- EM2 can have this fractional error rate before being called bad
                                                                0.1, #ZDC- EM3 can have this fractional error rate before being called bad
                                                                0.1, #ZDC- EM4 can have this fractional error rate before being called bad
                                                                0.1, #ZDC- EM5 can have this fractional error rate before being called bad
                                                                0.1, #ZDC- HAD1 can have this fractional error rate before being called bad
                                                                0.1, #ZDC- HAD2 can have this fractional error rate before being called bad
                                                                0.1, #ZDC- HAD3 can have this fractional error rate before being called bad
                                                                0.1  #ZDC- HAD4 can have this fractional error rate before being called bad
                                                               )
)
