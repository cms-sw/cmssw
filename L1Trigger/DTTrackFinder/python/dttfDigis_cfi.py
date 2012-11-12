import FWCore.ParameterSet.Config as cms

dttfDigis = cms.EDProducer("DTTrackFinder",
    OutOfTime_Filter = cms.untracked.bool(False),
    PT_Assignment_nbits_PhiB = cms.untracked.int32(10),
    BX_min = cms.untracked.int32(-9),
    Extrapolation_Filter = cms.untracked.int32(1),
    Open_LUTs = cms.untracked.bool(False),
    CSCStub_Source = cms.InputTag("csctfTrackDigis"),
    BX_max = cms.untracked.int32(7),
    EtaTrackFinder = cms.untracked.bool(True),
    CSC_Eta_Cancellation = cms.untracked.bool(False),
    Overlap = cms.untracked.bool(True),
    Extrapolation_nbits_PhiB = cms.untracked.int32(8),
    Extrapolation_nbits_Phi = cms.untracked.int32(8),
    Extrapolation_21 = cms.untracked.bool(False),
    PT_Assignment_nbits_Phi = cms.untracked.int32(12),
    Debug = cms.untracked.int32(0),
    PHI_Assignment_nbits_Phi = cms.untracked.int32(10),
    PHI_Assignment_nbits_PhiB = cms.untracked.int32(10),
    DTDigi_Source = cms.InputTag("dtTriggerPrimitiveDigis"),
    OutOfTime_Filter_Window = cms.untracked.int32(1)
)


