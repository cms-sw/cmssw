#
# FFTJet simple peak analyzer configuration
#
# I. Volobouev, July 17, 2012
#
import math
import FWCore.ParameterSet.Config as cms
from RecoJets.FFTJetProducers.fftjetcommon_cfi import *

fftpeakSimpleAnalyzer = cms.EDAnalyzer(
    "SimpleFFTPeakAnalyzer",
    #
    # Label for the clustering tree
    treeLabel = cms.InputTag("fftjetpatreco", "FFTJetPatternRecognition"),
    #
    # Name/title for the output root ntuple
    ntupleTitle = cms.string("FFT_Peaks"),
    #
    # Conversion factor from scale squared times peak magnitude to Pt.
    # Note that this factor depends on the grid used for pattern resolution.
    # The default value given here is correct for the default grid only.
    ptConversionFactor = cms.double(128*256/(4.0*math.pi)),
    #
    # Minimum and maximum scale to use. We will skip the first
    # and the last level of the default set of scales.
    minNtupleScale = cms.double(0.087*1.00001),
    maxNtupleScale = cms.double(0.6/1.00001),
    #
    # The initial set of scales used by the pattern recognition stage.
    # This is also the final set unless clustering tree construction
    # is adaptive. Needed here for reading back non-adaptive trees.
    InitialScales = fftjet_patreco_scales_50,
    #
    # Bandwidth ratio for the tree distance calculator
    etaToPhiBandwidthRatio = cms.double(1.0/fftjet_phi_to_eta_bw_ratio)
)
