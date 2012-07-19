import math
import FWCore.ParameterSet.Config as cms

from RecoJets.FFTJetProducers.fftjetcommon_cfi import *

# Dijet pattern recognition filter configuration
fftjetDijetFilter = cms.EDFilter(
    "FFTJetDijetFilter",
    #
    # Label for the input clustering tree (sparse or dense)
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
    InitialScales = fftjet_patreco_scales_3_at_050,
    #
    # Clustering tree distance functor
    TreeDistanceCalculator = fftjet_fixed_bandwidth_distance,
    #
    # The scale to work with. Clustering tree will find the closest
    # level whose scale is equal to or above the one given here.
    fixedScale = cms.double(0.5/1.0001),
    #
    # Conversion factor from scale squared times peak magnitude to Pt.
    # Note that this factor depends on the grid used for pattern resolution.
    # The default value given here is correct for the default grid only.
    ptConversionFactor = cms.double(128*256/(4.0*math.pi)),
    #
    # The cuts which must be passed for this event to continue along
    # the data processing chain
    min1to0PtRatio = cms.double(0.6),
    minDeltaPhi = cms.double(2.7),
    maxThirdJetFraction = cms.double(0.1),
    minPt0 = cms.double(1.0),
    minPt1 = cms.double(1.0),
    maxPeakEta = cms.double(fftjet_standard_eta_range)
)
