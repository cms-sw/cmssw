import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------------
# parametrization of MET x/y shift vs. sumEt
pfMEtSysShiftCorrParameters_2011runAvsSumEt_data = cms.PSet(
    px = cms.string("-3.365e-1 + 4.801e-3*sumEt"),
    py = cms.string("+2.578e-1 - 6.124e-3*sumEt")
)

pfMEtSysShiftCorrParameters_2011runAvsSumEt_mc = cms.PSet(
    px = cms.string("-9.389e-2 + 1.815e-4*sumEt"),
    py = cms.string("+1.571e-1 - 3.710e-3*sumEt")
)

pfMEtSysShiftCorrParameters_2011runBvsSumEt_data = cms.PSet(
    px = cms.string("-3.265e-1 + 5.162e-3*sumEt"),
    py = cms.string("-1.956e-2 - 6.299e-3*sumEt")
)

pfMEtSysShiftCorrParameters_2011runBvsSumEt_mc = cms.PSet(
    px = cms.string("-1.070e-1 + 9.587e-5*sumEt"),
    py = cms.string("-1.517e-2 - 3.357e-3*sumEt")
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# parametrization of MET x/y shift vs. Nvtx
pfMEtSysShiftCorrParameters_2011runAvsNvtx_data = cms.PSet(
    px = cms.string("+3.87339e-1 + 2.58294e-1*Nvtx"),
    py = cms.string("-7.83502e-1 - 2.88899e-1*Nvtx")
)

pfMEtSysShiftCorrParameters_2011runAvsNvtx_mc = cms.PSet(
    px = cms.string("-1.94451e-2 - 4.38986e-3*Nvtx"),
    py = cms.string("-4.31368e-1 - 1.90753e-1*Nvtx")
)

pfMEtSysShiftCorrParameters_2011runBvsNvtx_data = cms.PSet(
    px = cms.string("+6.64470e-1 + 2.71292e-1*Nvtx"),
    py = cms.string("-1.23999e0 - 3.18661e-1*Nvtx")
)

pfMEtSysShiftCorrParameters_2011runBvsNvtx_mc = cms.PSet(
    px = cms.string("-9.89706e-2 + 6.64796e-3*Nvtx"),
    py = cms.string("-5.32495e-1 - 1.82195e-1*Nvtx")
)

selectedVerticesForMEtCorr = cms.EDFilter("VertexSelector",
    src = cms.InputTag('offlinePrimaryVertices'),
    cut = cms.string("isValid & ndof >= 4 & chi2 > 0 & tracksSize > 0 & abs(z) < 24 & abs(position.Rho) < 2."),
    filter = cms.bool(False)                                          
)
#--------------------------------------------------------------------------------

pfMEtSysShiftCorr = cms.EDProducer("SysShiftMETcorrInputProducer",
    src = cms.InputTag('pfMet'), # "raw"/uncorrected PFMEt, needed to access sumEt
    srcVertices = cms.InputTag('selectedVerticesForMEtCorr'),                                   
    parameter = pfMEtSysShiftCorrParameters_2011runBvsNvtx_data
)                                     

pfMEtSysShiftCorrSequence = cms.Sequence(selectedVerticesForMEtCorr * pfMEtSysShiftCorr)
