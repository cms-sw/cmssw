#
# FFTJet pileup analyzer configuration. Default is the PFJet
# "MC calibration" mode. Here, we collect FFTJetPileupEstimator
# summaries and do not collect FFTJetPileupProcessor histograms.
#
# I. Volobouev, April 27, 2011
#
import FWCore.ParameterSet.Config as cms

fftjet_pileup_analyzer = cms.EDAnalyzer(
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
    # Output ntuple name/title
    ntupleName = cms.string("FFTJetPileupAnalyzer"),
    ntupleTitle = cms.string("FFTJetPileupAnalyzer ntuple"),
    #
    # Settings for the types of info we are collecting
    collectHistos = cms.bool(False),
    collectPileup = cms.bool(True),
    collectSummaries = cms.bool(True),
    verbosePileupInfo = cms.bool(False)
)
