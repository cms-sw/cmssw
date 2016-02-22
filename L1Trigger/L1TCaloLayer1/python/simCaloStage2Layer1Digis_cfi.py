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
    useLUT = cms.bool(True),
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
    hfSF30 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00),
    hfSF31 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00),
    hfSF32 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00),
    hfSF33 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00),
    hfSF34 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00),
    hfSF35 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00),
    hfSF36 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00),
    hfSF37 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00),
    hfSF38 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00),
    hfSF39 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00),
    hfSF40 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00),
    hfSF41 = cms.vdouble(1.00, 1.00, 1.00, 1.00, 1.00)
    )
