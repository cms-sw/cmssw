#
# FFTJet image recorder configuration
#
# I. Volobouev, Nov 5, 2012
#
import math
import FWCore.ParameterSet.Config as cms

from RecoJets.FFTJetProducers.fftjetcommon_cfi import *

fftjetImageRecorder = cms.EDAnalyzer(
    "FFTJetImageRecorder",
    #
    # The main eta and phi scale factors for the pattern recognition kernel
    kernelEtaScale = cms.double(math.sqrt(1.0/fftjet_phi_to_eta_bw_ratio)),
    kernelPhiScale = cms.double(math.sqrt(fftjet_phi_to_eta_bw_ratio)),
    #
    # Attempt to correct the jet finding efficiency near detector eta limits?
    fixEfficiency = cms.bool(False),
    #
    # Minimum and maximum eta bin number for 1d convolver. Also used
    # to indicate detector limits for 2d convolvers in case "fixEfficiency"
    # is True.
    convolverMinBin = cms.uint32(0),
    convolverMaxBin = cms.uint32(fftjet_large_int),
    #
    # Label for the produced objects (unused)
    outputLabel = cms.string("FFTJetImageRecorder"),
    #
    # Label for the input collection of Candidate objects
    src = cms.InputTag("towerMaker"),
    #
    # Label for the jets which will be produced. The algorithm might do
    # different things depending on the type. In particular, vertex
    # correction may be done for "CaloJet"
    jetType = cms.string("CaloJet"),
    #
    # Perform vertex correction?
    doPVCorrection = cms.bool(False),
    #
    # Label for the input collection of vertex objects. Meaningful
    # only when "doPVCorrection" is True
    srcPVs = cms.InputTag("offlinePrimaryVertices"),
    #
    # If this vector is empty, 2d convolver will be used.
    etaDependentScaleFactors = cms.vdouble(),
    #
    # Eta-dependent magnitude factors for the data. These can be used
    # to correct for various things (including the eta-dependent scale
    # factors above).
    etaDependentMagnutideFactors = cms.vdouble(),
    #
    # Configuration for the energy discretization grid. Don't forget
    # to modify "etConversionFactor" appropriately if you change this.
    GridConfiguration = fftjet_grid_256_128,
    #
    # Conversion factor to get from the grid values to Et.
    # This is number of bins in phi divided by bin width in eta.
    etConversionFactor = cms.double(128.0/(4.0*math.pi/256.0)),
    #
    # The set of scales to use
    InitialScales = fftjet_patreco_scales_50,
    #
    # Anomalous calo tower definition (comes from JetProducers default)
    anomalous = fftjet_anomalous_tower_default,
    #
    # The following has to be set to true
    insertCompleteEvent = cms.bool(True),
    #
    # Effective scale to use for the discretized event without smoothing
    completeEventScale = cms.double(fftjet_complete_event_scale),
    #
    # The power of the scale multiplied by the smoothed Et
    scalePower = cms.double(2.0)
)
