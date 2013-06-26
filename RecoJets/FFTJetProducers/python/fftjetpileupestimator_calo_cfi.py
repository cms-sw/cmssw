# MC-mix calibrated pile-up estimator for CaloJets

import FWCore.ParameterSet.Config as cms
import RecoJets.FFTJetProducers.mixed_calib_calo_ttbar_result as calib_calo

# FFTJet puleup estimator module configuration
fftjetPileupEstimatorCalo = cms.EDProducer(
    "FFTJetPileupEstimator",
    #
    # Label for the input info
    inputLabel = cms.InputTag("pileupprocessor", "FFTJetPileupCalo"),
    #
    # Label for the produced objects
    outputLabel = cms.string("FFTJetPileupEstimateCalo"),
    #
    # Conversion factor from the calibration curve value
    # to the mean transverse energy density
    ptToDensityFactor = cms.double(1.0),
    #
    # Fixed cdf value (optimized for CaloJets)
    cdfvalue = cms.double(0.4),
    #
    # Filter number (there is only one filter for production runs)
    filterNumber = cms.uint32(0),
    #
    # Uncertainty zones for the Neyman belt
    uncertaintyZones = cms.vdouble(calib_calo.uncertaintyZones),
    #
    # Calibration and uncertainty curves
    calibrationCurve = calib_calo.calibrationCurve,
    uncertaintyCurve = calib_calo.uncertaintyCurve,
    #
    # Parameters related to calibration curve access from DB
    calibTableRecord = cms.string("calibTableRecord"),
    calibTableCategory = cms.string("calibTableCategory"),
    uncertaintyZonesName = cms.string("uncertaintyZonesName"),
    calibrationCurveName = cms.string("calibrationCurveName"),
    uncertaintyCurveName = cms.string("uncertaintyCurveName"),
    loadCalibFromDB = cms.bool(False)
)
