import FWCore.ParameterSet.Config as cms

# Set the HLT paths
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOSiStripCalMinBiasAfterAbortGapHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
##     HLTPaths = [
##     #Minimum Bias
##     "HLT_MinBias*"
##     ],
    eventSetupPathsKey = 'SiStripCalMinBiasAfterAbortGapHI',
    throw = False # tolerate triggers stated above, but not available
    )

# Select only events where tracker had HV on (according to DCS bit information)
# AND respective partition is in the run (according to FED information)
import CalibTracker.SiStripCommon.SiStripDCSFilter_cfi
DCSStatusForSiStripCalMinBiasAAG = CalibTracker.SiStripCommon.SiStripDCSFilter_cfi.siStripDCSFilter.clone()

# Select pp-like events based on the pixel cluster multiplicity
#import HLTrigger.special.hltPixelActivityFilter_cfi
#HLTPixelActivityFilterForSiStripCalMinBias = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone()
#HLTPixelActivityFilterForSiStripCalMinBias.maxClusters = 500
#HLTPixelActivityFilterForSiStripCalMinBias.inputTag    = 'siPixelClusters'

# Select only good tracks
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOSiStripCalMinBiasAfterAbortGap = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()

ALCARECOSiStripCalMinBiasAfterAbortGap.filter         = True    ##do not store empty events	
ALCARECOSiStripCalMinBiasAfterAbortGap.src            = 'hiGeneralTracks'
ALCARECOSiStripCalMinBiasAfterAbortGap.applyBasicCuts = True
ALCARECOSiStripCalMinBiasAfterAbortGap.ptMin          = 0.8     ##GeV
ALCARECOSiStripCalMinBiasAfterAbortGap.nHitMin        = 6       ## at least 6 hits required
ALCARECOSiStripCalMinBiasAfterAbortGap.chi2nMax       = 10.

ALCARECOSiStripCalMinBiasAfterAbortGap.GlobalSelector.applyIsolationtest    = False
ALCARECOSiStripCalMinBiasAfterAbortGap.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOSiStripCalMinBiasAfterAbortGap.GlobalSelector.applyJetCountFilter   = False

ALCARECOSiStripCalMinBiasAfterAbortGap.TwoBodyDecaySelector.applyMassrangeFilter    = False
ALCARECOSiStripCalMinBiasAfterAbortGap.TwoBodyDecaySelector.applyChargeFilter       = False
ALCARECOSiStripCalMinBiasAfterAbortGap.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOSiStripCalMinBiasAfterAbortGap.TwoBodyDecaySelector.applyMissingETFilter    = False

# Sequence with the filter for the Pixel activity #
#seqALCARECOSiStripCalMinBias = cms.Sequence(ALCARECOSiStripCalMinBiasHLT*HLTPixelActivityFilterForSiStripCalMinBias*DCSStatusForSiStripCalMinBias*ALCARECOSiStripCalMinBias)

seqALCARECOSiStripCalMinBiasAfterAbortGap = cms.Sequence(ALCARECOSiStripCalMinBiasAfterAbortGapHLT *
                                                         DCSStatusForSiStripCalMinBiasAAG *
                                                         ALCARECOSiStripCalMinBiasAfterAbortGap)
