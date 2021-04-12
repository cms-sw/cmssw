import FWCore.ParameterSet.Config as cms

simDttfDigis = cms.EDProducer("DTTrackFinder",
    BX_max = cms.untracked.int32(7),
    BX_min = cms.untracked.int32(-9),
    CSCStub_Source = cms.InputTag("simCsctfTrackDigis"),
    CSC_Eta_Cancellation = cms.untracked.bool(False),
    DTDigi_Source = cms.InputTag("simDtTriggerPrimitiveDigis"),
    Debug = cms.untracked.int32(0),
    EtaTrackFinder = cms.untracked.bool(True),
    Extrapolation_21 = cms.untracked.bool(False),
    Extrapolation_Filter = cms.untracked.int32(1),
    Extrapolation_nbits_Phi = cms.untracked.int32(8),
    Extrapolation_nbits_PhiB = cms.untracked.int32(8),
    Open_LUTs = cms.untracked.bool(False),
    OutOfTime_Filter = cms.untracked.bool(False),
    OutOfTime_Filter_Window = cms.untracked.int32(1),
    Overlap = cms.untracked.bool(True),
    PHI_Assignment_nbits_Phi = cms.untracked.int32(10),
    PHI_Assignment_nbits_PhiB = cms.untracked.int32(10),
    PT_Assignment_nbits_Phi = cms.untracked.int32(12),
    PT_Assignment_nbits_PhiB = cms.untracked.int32(10)
)
