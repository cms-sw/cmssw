import math
import FWCore.ParameterSet.Config as cms

from RecoJets.FFTJetProducers.fftjetcommon_cfi import *

fftjet_default_recombination_scale = 0.5

# FFTJet jet producer configuration
fftjetJetMaker = cms.EDProducer(
    "FFTJetProducer",
    #
    # Label for the input clustering tree (must be sparse)
    treeLabel = cms.InputTag("fftjetpatreco", "FFTJetPatternRecognition"),
    #
    # Do we have the complete event at the lowest clustering tree scale?
    # Note that sparse clustering tree removes it by default, even if
    # it is inserted by the pattern recognition module.
    insertCompleteEvent = cms.bool(fftjet_insert_complete_event),
    completeEventScale = cms.double(fftjet_complete_event_scale),
    #
    # The initial set of scales used by the pattern recognition stage.
    # This is also the final set unless clustering tree construction
    # is adaptive. Needed here for reading back non-adaptive trees.
    InitialScales = fftjet_patreco_scales_50,
    #
    # Label for the produced objects
    outputLabel = cms.string("MadeByFFTJet"),
    #
    # Label for the input collection of Candidate objects
    src = cms.InputTag("towerMaker"),
    #
    # Type of the jets which will be produced (should be consistent with
    # the input collection). Valid types are "BasicJet", "GenJet", "CaloJet",
    # "PFJet", and "TrackJet". The algorithm might do different things
    # depending on the type. In particular, vertex correction may be done
    # for "CaloJet".
    jetType = cms.string("CaloJet"),
    #
    # Perform vertex correction?
    doPVCorrection = cms.bool(False),
    #
    # Label for the input collection of vertex objects. Meaningful
    # only when "doPVCorrection" is True
    srcPVs = cms.InputTag("offlinePrimaryVertices"),
    #
    # Anomalous calo tower definition (comes from RecoJets default)
    anomalous = fftjet_anomalous_tower_default,
    #
    # Magnitude correction factors (used only with gridded algorithms)
    etaDependentMagnutideFactors = cms.vdouble(),
    #
    # If a gridded algorithm is used, do we want to pick up the discretized
    # energy flow grid from the event record?
    reuseExistingGrid = cms.bool(False),
    #
    # If we do not reuse an existing grid, we need to provide
    # the grid configuration
    GridConfiguration = fftjet_grid_256_128,
    #
    # Maximum number of iterations allowed for the iterative jet
    # fitting. One-shot method is used if this number is 0 or 1.
    maxIterations = cms.uint32(1),
    #
    # Number of leading jets for which the iterative jet fitting must
    # converge before iterations are declared successful. This parameter
    # is not terribly meaningfule unless you know how many jets you expect
    # to get.
    nJetsRequiredToConverge = cms.uint32(10),
    #
    # The distance cutoff for the convergence. The distance between
    # the jets on two subsequent iterations must be less than this
    # cutoff in order to declare that the jet reconstruction has
    # converged. The distance function is defined by the "jetDistanceCalc"
    # parameter. Used only if "maxIterations" is larger than 1.
    convergenceDistance = cms.double(1.0e-6),
    #
    # Are we going to produce the set of constituents for each jet?
    # If we are not doing this, the code will run faster.
    assignConstituents = cms.bool(True),
    #
    # Are we going to resum constituents to calculate jet 4-vectors?
    # This only makes sense when a gridded algorithm is used in the
    # crisp 4-vector recombination mode to determine jet areas (note
    # that "recombinationDataCutoff" parameter should be negative),
    # and resumming is used to emulate vector algorithm recombination.
    resumConstituents = cms.bool(False),
    #
    # Noise sigma parameter for the background functor (the interface
    # to noise modeling is likely to be changed in the future)
    noiseLevel = cms.double(0.15),
    #
    # Number of clusters requested. Works with both "locallyAdaptive"
    # and "globallyAdaptive" resolution schemes.
    nClustersRequested = cms.uint32(4),
    #
    # Maximum eta for gridded recombination algorithms. Grid cells
    # with eta values ou t
    gridScanMaxEta = cms.double(fftjet_standard_eta_range),
    #
    # Are we going to use gridded or vector algorithm? Vector algoritms
    # are slightly more precise (no binning uncertainty introduced). However,
    # jet-by-jet jet areas can be calculated only by gridded algorithms.
    useGriddedAlgorithm = cms.bool(False),
    #
    # The recombination algorithm used. For vector algorithms, possible
    # specifications are:
    #   "Kernel"     -- use 4-vector recombination scheme
    #   "EtCentroid" -- use Et centroid (or "original Snowmass") scheme
    #   "EtSum"      -- set the jet direction to the precluster direction
    # For gridded algorithms additional specifications are available:
    # "FasterKernel", "FasterEtCentroid", and "FasterEtSum". See the
    # comments in the "FasterKernelRecombinationAlg.hh" header of the
    # FFTJet package for limitations of those faster algorithms.
    recombinationAlgorithm = cms.string("Kernel"),
    #
    # Are we going to utilize crisp or fuzzy clustering?
    isCrisp = cms.bool(True),
    #
    # A parameter which defines when we will attempt to split the energy
    # of a calorimeter cell if it is unlikely to belong to any jet and
    # to the noise. Works with Et-dependent membership functions only.
    # The default value of 0 means don't split, just assign this energy
    # deposition to the unclustered energy.
    unlikelyBgWeight = cms.double(0.0),
    #
    # The data cutoff for the gridded algorithms. Set this cutoff
    # to some negative number if you want to calculate jet areas
    # (this can also be done by turning on pile-up calculation
    # as a separate step.) Set it to 0 or some positive number
    # if you want to improve the code speed.
    recombinationDataCutoff = cms.double(0.0),
    #
    # The built-in precluster selection for subsequent jet reconstruction
    # can be performed according to the following schemes which, basically,
    # describe how the resolution of the Gaussian filter is chosen:
    #  "fixed"            -- use the same user-selected resolution across
    #                        the whole eta-phi space
    #  "maximallyStable"  -- pick up a single resolution according to
    #                        a jet configuration stability criterion
    #  "globallyAdaptive" -- pick up a single resolution which gives
    #                        a desired number of jets
    #  "locallyAdaptive"  -- use different resolutions in different parts
    #                        of the eta-phi space in order to maximize
    #                        a certain optimization criterion
    resolution = cms.string("fixed"),
    #
    # Scale parameter for the "fixed" and "locallyAdaptive" resolution schemes
    fixedScale = cms.double(0.15),
    #
    # Minimum and maximum stable scales for the "maximallyStable"
    # resolution scheme. Value of 0 means there is no limit, and
    # all scales in the clustering tree are considered.
    minStableScale = cms.double(0.0),
    maxStableScale = cms.double(0.0),
    #
    # Stability exponent for the "maximallyStable" resolution scheme
    stabilityAlpha = cms.double(0.5),
    #
    # The precluster discriminator which works together with the
    # resolution selection scheme
    PeakSelectorConfiguration = cms.PSet(
        Class = cms.string("SimplePeakSelector"),
        magCut = cms.double(0.1),
        driftSpeedCut = cms.double(1.0e100),
        magSpeedCut = cms.double(-1.0e100),
        lifeTimeCut = cms.double(-1.0e100),
        NNDCut = cms.double(-1.0e100),
        etaCut = cms.double(1.0e100)
    ),
    #
    # The jet membership function
    jetMembershipFunction = fftjet_jet_membership_cone,
    #
    # The noise membership function
    bgMembershipFunction = fftjet_noise_membership_smallconst,
    #
    # The recombination scale function
    recoScaleCalcPeak = cms.PSet(
        Class = cms.string("ConstDouble"),
        value = cms.double(fftjet_default_recombination_scale)
    ),
    #
    # The function which calculates eta-to-phi bandwidth ratio
    # for the jet membership function. If the ratio is set to 0,
    # the "setScaleRatio" membership function method will never
    # be called, and the default ratio built into the membership
    # functionwill be used instead.
    recoScaleRatioCalcPeak = fftjet_peakfunctor_const_zero,
    #
    # The function which calculates the factor to be multiplied by
    # the membership function
    memberFactorCalcPeak = fftjet_peakfunctor_const_one,
    #
    # The following parameters must be specified if "maxIterations" value
    # is larger than 1. They are used in the iterative mode only.
    #  recoScaleCalcJet = ,
    #  recoScaleRatioCalcJet = ,
    #  memberFactorCalcJet = ,
    #  jetDistanceCalc = ,
    #
    recoScaleCalcJet = cms.PSet(
        Class = cms.string("ConstDouble"),
        value = cms.double(fftjet_default_recombination_scale)
    ),
    recoScaleRatioCalcJet = fftjet_peakfunctor_const_one,
    memberFactorCalcJet = fftjet_peakfunctor_const_one,
    jetDistanceCalc = fftjet_convergence_jet_distance,
    #
    # Are we going to estimate the pile-up using actual jet shapes?
    # Note that the following _must_ be defined if we want to do this:
    #   recoScaleCalcJet, recoScaleRatioCalcJet, memberFactorCalcJet,
    #   PileupGridConfiguration, and pileupDensityCalc
    calculatePileup = cms.bool(False),
    #
    # If the pile-up is estimated, do we want to subtract it?
    subtractPileup = cms.bool(False),
    #
    # If the pile-up is both estimated and subtracted, do we want to use
    # the 4-vector pile-up subtraction scheme? (The alternative is based
    # on scaling the jet Pt).
    subtractPileupAs4Vec = cms.bool(False),
    #
    # Source of the pile-up energy flow data
    pileupLabel = cms.InputTag("pileupestimator", "FFTJetPileupEstimatePF"),
    #
    # Label for GenJet collection in the "fromGenJets" resolution mode
    genJetsLabel = cms.InputTag("fftgenjetproducer", "MadeByFFTJet"),
    #
    # Max number of preclusters. Does not take into account the possibility
    # of further precluster removal by setting its membership factor to 0.
    maxInitialPreclusters = cms.uint32(2147483647),
    #
    # Parameters related to pileup shape fetching from DB
    pileupTableRecord = cms.string("pileupTableRecord"),
    pileupTableName = cms.string("pileupTableName"),
    pileupTableCategory = cms.string("pileupTableCategory"),
    loadPileupFromDB = cms.bool(False)
)
