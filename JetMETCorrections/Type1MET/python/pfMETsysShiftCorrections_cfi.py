import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------------
# parametrization of MET x/y shift vs. sumEt
pfMEtSysShiftCorrParameters_2011runAvsSumEt_data = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("-3.365e-1 + 4.801e-3*sumEt"),
    py = cms.string("+2.578e-1 - 6.124e-3*sumEt")
))

pfMEtSysShiftCorrParameters_2011runAvsSumEt_mc = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("-9.389e-2 + 1.815e-4*sumEt"),
    py = cms.string("+1.571e-1 - 3.710e-3*sumEt")
))

pfMEtSysShiftCorrParameters_2011runBvsSumEt_data = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("-3.265e-1 + 5.162e-3*sumEt"),
    py = cms.string("-1.956e-2 - 6.299e-3*sumEt")
))

pfMEtSysShiftCorrParameters_2011runBvsSumEt_mc = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("-1.070e-1 + 9.587e-5*sumEt"),
    py = cms.string("-1.517e-2 - 3.357e-3*sumEt")
))

pfMEtSysShiftCorrParameters_2011runAplusBvsSumEt_data = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("-5.65217e-01 + 5.42436e-03*sumEt"),
    py = cms.string("+4.54054e-01 - 6.73607e-03*sumEt")
))

pfMEtSysShiftCorrParameters_2011runAplusBvsSumEt_mc = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("-4.53909e-02 - 2.55863e-05*sumEt"),
    py = cms.string("+1.27947e-01 - 3.62604e-03*sumEt")    
))

pfMEtSysShiftCorrParameters_2012runAvsSumEt_data = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("-7.67892e-01 + 5.76983e-03*sumEt"),
    py = cms.string("+5.54005e-01 - 2.94046e-03*sumEt")
))

pfMEtSysShiftCorrParameters_2012runAvsSumEt_mc = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("+1.77344e-01 - 1.34333e-03*sumEt"),
    py = cms.string("+8.08402e-01 - 2.84264e-03*sumEt")
))
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# parametrization of MET x/y shift vs. Nvtx
pfMEtSysShiftCorrParameters_2011runAvsNvtx_data = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("+3.87339e-1 + 2.58294e-1*Nvtx"),
    py = cms.string("-7.83502e-1 - 2.88899e-1*Nvtx")
))

pfMEtSysShiftCorrParameters_2011runAvsNvtx_mc = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("-1.94451e-2 - 4.38986e-3*Nvtx"),
    py = cms.string("-4.31368e-1 - 1.90753e-1*Nvtx")
))

pfMEtSysShiftCorrParameters_2011runBvsNvtx_data = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("+6.64470e-1 + 2.71292e-1*Nvtx"),
    py = cms.string("-1.23999e0 - 3.18661e-1*Nvtx")
))

pfMEtSysShiftCorrParameters_2011runBvsNvtx_mc = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("-9.89706e-2 + 6.64796e-3*Nvtx"),
    py = cms.string("-5.32495e-1 - 1.82195e-1*Nvtx")
))

pfMEtSysShiftCorrParameters_2011runAplusBvsNvtx_data = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("+3.64118e-01 + 2.93853e-01*Nvtx"),
    py = cms.string("-7.17757e-01 - 3.57309e-01*Nvtx")
))

pfMEtSysShiftCorrParameters_2011runAplusBvsNvtx_mc = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("-4.79178e-02 + 8.62653e-04*Nvtx"),
    py = cms.string("-4.54408e-01 - 1.89684e-01*Nvtx")
))

pfMEtSysShiftCorrParameters_2012runAplusBvsNvtx_data = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("+1.68804e-01 + 3.37139e-01*Nvtx"),
    py = cms.string("-1.72555e-01 - 1.79594e-01*Nvtx")
))

pfMEtSysShiftCorrParameters_2012runAplusBvsNvtx_mc = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("+2.22335e-02 - 6.59183e-02*Nvtx"),
    py = cms.string("+1.52720e-01 - 1.28052e-01*Nvtx")
))

pfMEtSysShiftCorrParameters_2012runABCvsNvtx_data = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("+0.2661 + 0.3217*Nvtx"),
    py = cms.string("-0.2251 - 0.1747*Nvtx")
))

pfMEtSysShiftCorrParameters_2012runABCvsNvtx_mc = cms.VPSet(cms.PSet(
    numJetsMin = cms.int32(-1),
    numJetsMax = cms.int32(-1),
    px = cms.string("+0.1166 + 0.0200*Nvtx"),
    py = cms.string("+0.2764 - 0.1280*Nvtx")
))

selectedVerticesForMEtCorr = cms.EDFilter("VertexSelector",
    src = cms.InputTag('offlinePrimaryVertices'),
    cut = cms.string("isValid & ndof >= 4 & chi2 > 0 & tracksSize > 0 & abs(z) < 24 & abs(position.Rho) < 2."),
    filter = cms.bool(False)                                          
)
#--------------------------------------------------------------------------------

pfMEtSysShiftCorr = cms.EDProducer("SysShiftMETcorrInputProducer",
    srcMEt = cms.InputTag('pfMet'), # "raw"/uncorrected PFMEt, needed to access sumEt
    srcVertices = cms.InputTag('selectedVerticesForMEtCorr'),
    srcJets = cms.InputTag('ak5PFJets'),
    jetPtThreshold = cms.double(10.),
    parameter = pfMEtSysShiftCorrParameters_2012runAplusBvsNvtx_data
)                                     

pfMEtSysShiftCorrSequence = cms.Sequence(selectedVerticesForMEtCorr * pfMEtSysShiftCorr)
