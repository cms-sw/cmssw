# Uncalibrated pile-up estimator for PFJets

import math
import FWCore.ParameterSet.Config as cms

dummy_functor = cms.PSet(
    Class = cms.string("Polynomial"),
    c0 = cms.double(-1.0)
)

# FFTJet puleup estimator module configuration
fftjetPileupEstimatorPf_uncal = cms.EDProducer(
    "FFTJetPileupEstimator",
    #
    # Label for the input info
    inputLabel = cms.InputTag("pileupprocessor", "FFTJetPileupPF"),
    #
    # Label for the produced objects
    outputLabel = cms.string("FFTJetPileupEstimatePFUncalib"),
    #
    # Conversion factor from total pile-up pt to its density
    # (don't care for uncalibrated run)
    ptToDensityFactor = cms.double(1.0),
    #
    # Fixed cdf value (optimized for PFJets)
    cdfvalue = cms.double(0.5),
    #
    # Filter number (there is only one filter for production runs)
    filterNumber = cms.uint32(0),
    #
    # Uncertainty zones for the Neyman belt (don't care for uncalibrated run)
    uncertaintyZones = cms.vdouble(),
    #
    # Calibration and uncertainty curves (don't care for uncalibrated run)
    calibrationCurve = dummy_functor,
    uncertaintyCurve = dummy_functor,
    #
    # Parameters related to calibration curve access from DB
    calibTableRecord = cms.string("calibTableRecord"),
    calibTableCategory = cms.string("calibTableCategory"),
    uncertaintyZonesName = cms.string("uncertaintyZonesName"),
    calibrationCurveName = cms.string("calibrationCurveName"),
    uncertaintyCurveName = cms.string("uncertaintyCurveName"),
    loadCalibFromDB = cms.bool(False)
)
