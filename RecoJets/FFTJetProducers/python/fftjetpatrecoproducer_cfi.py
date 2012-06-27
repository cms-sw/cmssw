import math
import FWCore.ParameterSet.Config as cms

from RecoJets.FFTJetProducers.fftjetcommon_cfi import *

# FFTJet pattern recognition module configuration
fftjetPatrecoProducer = cms.EDProducer(
    "FFTJetPatRecoProducer",
    #
    # The main eta and phi scale factors for the pattern recognition kernel
    kernelEtaScale = cms.double(math.sqrt(1.0/fftjet_phi_to_eta_bw_ratio)),
    kernelPhiScale = cms.double(math.sqrt(fftjet_phi_to_eta_bw_ratio)),
    #
    # Make the clustering trees? If you do not make the trees,
    # you should at least turn on the "storeDiscretizationGrid"
    # flag, otherwise this module will not produce anything at all.
    makeClusteringTree = cms.bool(True),
    #
    # Verify data conversion? For trees, this is only meaningful with
    # double precision storage. Grids, however, will always be verified
    # if this flag is set.
    verifyDataConversion = cms.untracked.bool(False),
    #
    # Are we going to produce sparse or full clustering trees
    sparsify = cms.bool(True),
    #
    # Are we going to store the discretized energy flow?
    storeDiscretizationGrid = cms.bool(False),
    #
    # Are we going to dump discretized energy flow into an external file?
    # Empty file name means "no".
    externalGridFile = cms.string(""),
    #
    # Configuration for the preliminary peak finder.
    # Its main purpose is to reject peaks produced by the FFT round-off noise.
    peakFinderMaxEta = cms.double(fftjet_standard_eta_range),
    peakFinderMaxMagnitude = cms.double(1.e-8),
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
    # Insert complete event at the end when the clustering tree is constructed?
    insertCompleteEvent = cms.bool(fftjet_insert_complete_event),
    #
    # The scale variable for the complete event. Should be smaller than
    # any other pattern recognition scale but not too small so that the
    # tree can be nicely visualized in the ln(scale) space.
    completeEventScale = cms.double(fftjet_complete_event_scale),
    #
    # The grid data cutoff for the complete event
    completeEventDataCutoff = cms.double(0.0),
    #
    # Label for the produced objects
    outputLabel = cms.string("FFTJetPatternRecognition"),
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
    # Are we going to perform adaptive clustering? Setting the maximum
    # number of adaptive scales to 0 turns adaptive clustering off.
    maxAdaptiveScales = cms.uint32(0),
    #
    # Minimum distance between the scales (in the ln(scale) space)
    # for adaptive clustering. Meaningful only when the "maxAdaptiveScales"
    # parameter is not 0.
    minAdaptiveRatioLog = cms.double(0.01),
    #
    # Eta-dependent scale factors for the sequential 1d convolver.
    # If this vector is empty, 2d convolver will be used.
    etaDependentScaleFactors = cms.vdouble(),
    #
    # Eta-dependent magnitude factors for the data. These can be used
    # to correct for various things (including the eta-dependent scale
    # factors above).
    etaDependentMagnutideFactors = cms.vdouble(),
    #
    # Configuration for the energy discretization grid
    GridConfiguration = fftjet_grid_256_128,
    #
    # Configuration for the peak selector determining which peaks
    # are kept when the clustering tree is constructed
    PeakSelectorConfiguration = fftjet_peak_selector_allpass,
    #
    # The initial set of scales used by the pattern recognition stage.
    # This is also the final set unless clustering tree construction
    # is adaptive.
    InitialScales = fftjet_patreco_scales_50,
    #
    # Configuration for the clustering tree sparsification.
    # 
    # Do not write the last tree level (the complete event) into the sparse
    # tree. This is done by setting the "maxLevelNumber" parameter to -1
    # in which case the counting for the max level is performed backwards
    # from the last level. Counting backwards is especially useful in the
    # adaptive clustering mode when the number of clustering tree levels
    # is not known in advance.
    SparsifierConfiguration = cms.PSet(
        maxLevelNumber = cms.int32(-1),
        filterMask = cms.uint32(fftjet_large_int),
        userScales = cms.vdouble()
    ),
    #
    # Clustering tree distance functor
    TreeDistanceCalculator = fftjet_fixed_bandwidth_distance,
    #
    # Anomalous calo tower definition (comes from JetProducers default)
    anomalous = fftjet_anomalous_tower_default
)
