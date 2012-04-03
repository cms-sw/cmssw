import FWCore.ParameterSet.Config as cms

pfMEtSysShiftCorrParameters_2011runA_data = cms.PSet(
    px = cms.string("-3.365e-1 + 4.801e-3*sumEt"),
    py = cms.string("+2.578e-1 - 6.124e-3*sumEt")
)

pfMEtSysShiftCorrParameters_2011runA_mc = cms.PSet(
    px = cms.string("-9.389e-2 + 1.815e-4*sumEt"),
    py = cms.string("+1.571e-1 - 3.710e-3*sumEt")
)

pfMEtSysShiftCorrParameters_2011runB_data = cms.PSet(
    px = cms.string("-3.265e-1 + 5.162e-3*sumEt"),
    py = cms.string("-1.956e-2 - 6.299e-3*sumEt")
)

pfMEtSysShiftCorrParameters_2011runB_mc = cms.PSet(
    px = cms.string("-1.070e-1 + 9.587e-5*sumEt"),
    py = cms.string("-1.517e-2 - 3.357e-3*sumEt")
)

pfMEtSysShiftCorr = cms.EDProducer("SysShiftMETcorrInputProducer",
    src = cms.InputTag('pfMet'), # "raw"/uncorrected PFMEt, needed to access sumEt                                     
    parameter = pfMEtSysShiftCorrParameters_2011runB_data
)                                     
