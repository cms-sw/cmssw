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

pfMEtSysShiftCorrParameters_2011runAplusBvsSumEt_data = cms.PSet(
    px = cms.string("-5.65217e-01 + 5.42436e-03*sumEt"),
    py = cms.string("+4.54054e-01 - 6.73607e-03*sumEt")
)

pfMEtSysShiftCorrParameters_2011runAplusBvsSumEt_mc = cms.PSet(
    px = cms.string("-4.53909e-02 - 2.55863e-05*sumEt"),
    py = cms.string("+1.27947e-01 - 3.62604e-03*sumEt")    
)

pfMEtSysShiftCorrParameters_2012runAvsSumEt_data = cms.PSet(
    px = cms.string("-7.67892e-01 + 5.76983e-03*sumEt"),
    py = cms.string("+5.54005e-01 - 2.94046e-03*sumEt")
)

pfMEtSysShiftCorrParameters_2012runAvsSumEt_mc = cms.PSet(
    px = cms.string("+1.77344e-01 - 1.34333e-03*sumEt"),
    py = cms.string("+8.08402e-01 - 2.84264e-03*sumEt")
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

pfMEtSysShiftCorrParameters_2011runAplusBvsNvtx_data = cms.PSet(
    px = cms.string("+3.64118e-01 + 2.93853e-01*Nvtx"),
    py = cms.string("-7.17757e-01 - 3.57309e-01*Nvtx")
)

pfMEtSysShiftCorrParameters_2011runAplusBvsNvtx_mc = cms.PSet(
    px = cms.string("-4.79178e-02 + 8.62653e-04*Nvtx"),
    py = cms.string("-4.54408e-01 - 1.89684e-01*Nvtx")
)

pfMEtSysShiftCorrParameters_2012runAvsNvtx_data = cms.PSet(
    px = cms.string("+3.54233e-01 + 2.65299e-01*Nvtx"),
    py = cms.string("+1.88923e-01 - 1.66425e-01*Nvtx")
)

pfMEtSysShiftCorrParameters_2012runAvsNvtx_mc = cms.PSet(
    px = cms.string("-2.99576e-02 - 6.61932e-02*Nvtx"),
    py = cms.string("+3.70819e-01 - 1.48617e-01*Nvtx")
)

pfMEtSysShiftCorrParameters_2012runAplusBvsNvtx_data = cms.PSet(
    px = cms.string("+1.68804e-01 + 3.37139e-01*Nvtx"),
    py = cms.string("-1.72555e-01 - 1.79594e-01*Nvtx")
)

pfMEtSysShiftCorrParameters_2012runAplusBvsNvtx_mc = cms.PSet(
    px = cms.string("+2.22335e-02 - 6.59183e-02*Nvtx"),
    py = cms.string("+1.52720e-01 - 1.28052e-01*Nvtx")
)

pfMEtSysShiftCorrParameters_2012runABCDvsNvtx_data = cms.PSet( # CV: ReReco data + Summer'13 JEC
    px = cms.string("+4.83642e-02 + 2.48870e-01*Nvtx"),
    py = cms.string("-1.50135e-01 - 8.27917e-02*Nvtx")
)

pfMEtSysShiftCorrParameters_2012runABCDvsNvtx_mc = cms.PSet( # CV: Summer'12 MC + Summer'13 JEC
    px = cms.string("+1.62861e-01 - 2.38517e-02*Nvtx"),
    py = cms.string("+3.60860e-01 - 1.30335e-01*Nvtx")
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
    parameter = pfMEtSysShiftCorrParameters_2012runABCDvsNvtx_data
)                                     

pfMEtSysShiftCorrSequence = cms.Sequence(selectedVerticesForMEtCorr * pfMEtSysShiftCorr)
