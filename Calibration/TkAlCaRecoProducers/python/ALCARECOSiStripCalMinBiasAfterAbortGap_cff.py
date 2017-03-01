import FWCore.ParameterSet.Config as cms

# Set the HLT paths
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOSiStripCalMinBiasAfterAbortGapHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
##     HLTPaths = [
##     #Minimum Bias
##     "HLT_MinBias*"
##     ],
    eventSetupPathsKey = 'SiStripCalMinBiasAfterAbortGap',
    throw = False # tolerate triggers stated above, but not available
    )

# Select only events where tracker had HV on (according to DCS bit information)
# AND respective partition is in the run (according to FED information)
import CalibTracker.SiStripCommon.SiStripDCSFilter_cfi
DCSStatusForSiStripCalMinBiasAfterAbortGap = CalibTracker.SiStripCommon.SiStripDCSFilter_cfi.siStripDCSFilter.clone()

# Select only good tracks
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOSiStripCalMinBiasAfterAbortGap = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()

ALCARECOSiStripCalMinBiasAfterAbortGap.filter         = True   ##do not store empty events	
ALCARECOSiStripCalMinBiasAfterAbortGap.src            = 'generalTracks'
ALCARECOSiStripCalMinBiasAfterAbortGap.applyBasicCuts = True
ALCARECOSiStripCalMinBiasAfterAbortGap.ptMin          = 0.8    ##GeV
ALCARECOSiStripCalMinBiasAfterAbortGap.nHitMin        = 6      ## at least 6 hits required
ALCARECOSiStripCalMinBiasAfterAbortGap.chi2nMax       = 10.

ALCARECOSiStripCalMinBiasAfterAbortGap.GlobalSelector.applyIsolationtest    = False
ALCARECOSiStripCalMinBiasAfterAbortGap.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOSiStripCalMinBiasAfterAbortGap.GlobalSelector.applyJetCountFilter   = False

ALCARECOSiStripCalMinBiasAfterAbortGap.TwoBodyDecaySelector.applyMassrangeFilter    = False
ALCARECOSiStripCalMinBiasAfterAbortGap.TwoBodyDecaySelector.applyChargeFilter       = False
ALCARECOSiStripCalMinBiasAfterAbortGap.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOSiStripCalMinBiasAfterAbortGap.TwoBodyDecaySelector.applyMissingETFilter    = False

# Sequence #
seqALCARECOSiStripCalMinBiasAfterAbortGap = cms.Sequence(ALCARECOSiStripCalMinBiasAfterAbortGapHLT*
                                                         DCSStatusForSiStripCalMinBiasAfterAbortGap *
                                                         ALCARECOSiStripCalMinBiasAfterAbortGap)
