# MC-calibrated pile-up estimator for CaloJets

import FWCore.ParameterSet.Config as cms
import RecoJets.FFTJetProducers.pileup_calib_calo_cfi as calib_calo

pileup_estimator_eta_range = 5.0

# FFTJet puleup estimator module configuration
fftjet_pileup_estimator_calo = cms.EDProducer(
    "FFTJetPileupEstimator",
    #
    # Label for the input info
    inputLabel = cms.InputTag("pileupprocessor", "FFTJetPileupCalo"),
    #
    # Label for the produced objects
    outputLabel = cms.string("FFTJetPileupEstimateCalo"),
    #
    # Conversion factor from total pile-up Pt to its density.
    # The constant in the numerator is 1/(4 Pi)
    ptToDensityFactor = cms.double(0.07957747/pileup_estimator_eta_range),
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
    uncertaintyCurve = calib_calo.uncertaintyCurve
)
