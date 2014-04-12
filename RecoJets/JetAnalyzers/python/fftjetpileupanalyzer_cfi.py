#
# FFTJet pileup analyzer configuration. Default is the PFJet
# "MC calibration" mode. Here, we collect FFTJetPileupEstimator
# summaries and do not collect FFTJetPileupProcessor histograms.
#
# I. Volobouev, April 27, 2011
#
import math
import FWCore.ParameterSet.Config as cms

fftjetPileupAnalyzer = cms.EDAnalyzer(
    "FFTJetPileupAnalyzer",
    #
    # Label for the histograms produced by FFTJetPileupProcessor
    histoLabel = cms.InputTag("pileupprocessor", "FFTJetPileupPFStudy"),
    #
    # Label for the summary produced by FFTJetPileupEstimator
    summaryLabel = cms.InputTag("pileupestimator", "FFTJetPileupEstimatePFUncalib"),
    #
    # Label for the MC pileup summary
    pileupLabel = cms.string("addPileupInfo"),
    #
    # Labels for fastjet rho and sigma
    fastJetRhoLabel = cms.InputTag(""),
    fastJetSigmaLabel = cms.InputTag(""),
    #
    # Label for the energy discretization grid
    gridLabel = cms.InputTag("fftjetpatreco", "FFTJetPatternRecognition"),
    #
    # Label for the collection of primary vertices
    srcPVs = cms.InputTag("offlinePrimaryVertices"),
    #
    # Cut on the nDoF of the primary vertices
    vertexNdofCut = cms.double(4.0),
    #
    # Output ntuple name/title
    ntupleName = cms.string("FFTJetPileupAnalyzer"),
    ntupleTitle = cms.string("FFTJetPileupAnalyzer ntuple"),
    #
    # Settings for the types of info we are collecting
    collectHistos = cms.bool(False),
    collectPileup = cms.bool(True),
    collectOOTPileup = cms.bool(False),
    collectNumInteractions = cms.bool(True),
    collectFastJetRho = cms.bool(False),
    collectSummaries = cms.bool(True),
    collectGrids = cms.bool(False),
    collectGridDensity = cms.bool(False),
    collectVertexInfo = cms.bool(False),
    verbosePileupInfo = cms.bool(False),
    #
    # There is a bug somewhere in the module which builds
    # PileupSummaryInfo (it shows up in some events with OOP pileup).
    # The following kind-of helps avoiding crazy energy values.
    crazyEnergyCut = cms.double(2500.0)
)
