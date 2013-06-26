# MC-mix calibrated pile-up estimator for PFJets

import FWCore.ParameterSet.Config as cms
import RecoJets.FFTJetProducers.mixed_calib_pf_ttbar_result as calib_pf

# FFTJet puleup estimator module configuration
fftjetPileupEstimatorPf = cms.EDProducer(
    "FFTJetPileupEstimator",
    #
    # Label for the input info
    inputLabel = cms.InputTag("pileupprocessor", "FFTJetPileupPF"),
    #
    # Label for the produced objects
    outputLabel = cms.string("FFTJetPileupEstimatePF"),
    #
    # Conversion factor from the calibration curve value
    # to the mean transverse energy density
    ptToDensityFactor = cms.double(1.0),
    #
    # Fixed cdf value (optimized for PFJets)
    cdfvalue = cms.double(0.5),
    #
    # Filter number (there is only one filter for production runs)
    filterNumber = cms.uint32(0),
    #
    # Uncertainty zones for the Neyman belt (don't care for uncalibrated run)
    uncertaintyZones = cms.vdouble(calib_pf.uncertaintyZones),
    #
    # Calibration and uncertainty curves (don't care for uncalibrated run)
    calibrationCurve = calib_pf.calibrationCurve,
    uncertaintyCurve = calib_pf.uncertaintyCurve,
    #
    # Parameters related to calibration curve access from DB
    calibTableRecord = cms.string("calibTableRecord"),
    calibTableCategory = cms.string("calibTableCategory"),
    uncertaintyZonesName = cms.string("uncertaintyZonesName"),
    calibrationCurveName = cms.string("calibrationCurveName"),
    uncertaintyCurveName = cms.string("uncertaintyCurveName"),
    loadCalibFromDB = cms.bool(False)
)
