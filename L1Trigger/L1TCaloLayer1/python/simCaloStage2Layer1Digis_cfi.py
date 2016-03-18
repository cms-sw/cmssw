#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms

simCaloStage2Layer1Digis = cms.EDProducer(
    'L1TCaloLayer1',
    ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    hcalToken = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    useLSB = cms.bool(True),
    useECALLUT = cms.bool(True),
    useHCALLUT = cms.bool(True),
    useHFLUT = cms.bool(True),
    verbose = cms.bool(False),
    # Note that hf scale factors are temporarily placed here
    # These are used to compute HF look-up-tables
    # There are 12 eta bins in HF
    # The etCode is 8-bit which is used to look-up calibrated ET
    # The ET bins in which scale factors are derived are set in hfSFETBins
    # There should be as many scale factors as there are ETBins for each eta bin
    # hfLUT[etaBin][etCode] = etCode * hfSF[etaBin][etBin];
    # For ET codes below lowest ETBin, the first value is used
    # The maximum ET code is 255 - so, highest number below should be 256
    hfSFETBins = cms.vuint32(5, 20, 30, 50, 256),
    hfSF30 = cms.vdouble(1.00, 1.55, 1.60, 1.56, 1.46),
    hfSF31 = cms.vdouble(1.00, 1.49, 1.51, 1.41, 1.30),
    hfSF32 = cms.vdouble(1.00, 1.35, 1.38, 1.34, 1.29),
    hfSF33 = cms.vdouble(1.00, 1.29, 1.38, 1.42, 1.43),
    hfSF34 = cms.vdouble(1.00, 1.30, 1.44, 1.44, 1.42),
    hfSF35 = cms.vdouble(1.00, 1.42, 1.56, 1.52, 1.49),
    hfSF36 = cms.vdouble(1.00, 1.49, 1.60, 1.57, 1.52),
    hfSF37 = cms.vdouble(1.00, 1.59, 1.67, 1.63, 1.59),
    hfSF38 = cms.vdouble(1.00, 1.74, 1.83, 1.73, 1.69),
    hfSF39 = cms.vdouble(1.00, 1.86, 2.02, 1.94, 1.87),
    hfSF40 = cms.vdouble(1.00, 2.18, 2.66, 2.64, 2.49),
    hfSF41 = cms.vdouble(1.00, 2.43, 2.79, 2.64, 2.66)
    )
