import FWCore.ParameterSet.Config as cms

# Set the HLT paths
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOSiStripCalMinBiasHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
##     HLTPaths = [
##     #Minimum Bias
##     "HLT_MinBias*"
##     ],
    eventSetupPathsKey = 'SiStripCalMinBias',
    throw = False # tolerate triggers stated above, but not available
    )

# Select only events where tracker had HV on (according to DCS bit information)
# AND respective partition is in the run (according to FED information)
import CalibTracker.SiStripCommon.SiStripDCSFilter_cfi
DCSStatusForSiStripCalMinBias = CalibTracker.SiStripCommon.SiStripDCSFilter_cfi.siStripDCSFilter.clone()

# Select only good tracks
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOSiStripCalMinBias = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()

ALCARECOSiStripCalMinBias.filter         = True ##do not store empty events	
ALCARECOSiStripCalMinBias.src            = 'generalTracks'
ALCARECOSiStripCalMinBias.applyBasicCuts = True
ALCARECOSiStripCalMinBias.ptMin          = 0.8 ##GeV
ALCARECOSiStripCalMinBias.nHitMin        = 6 ## at least 6 hits required
ALCARECOSiStripCalMinBias.chi2nMax       = 10.

ALCARECOSiStripCalMinBias.GlobalSelector.applyIsolationtest    = False
ALCARECOSiStripCalMinBias.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOSiStripCalMinBias.GlobalSelector.applyJetCountFilter   = False

ALCARECOSiStripCalMinBias.TwoBodyDecaySelector.applyMassrangeFilter    = False
ALCARECOSiStripCalMinBias.TwoBodyDecaySelector.applyChargeFilter       = False
ALCARECOSiStripCalMinBias.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOSiStripCalMinBias.TwoBodyDecaySelector.applyMissingETFilter    = False

# Sequence #
seqALCARECOSiStripCalMinBias = cms.Sequence(ALCARECOSiStripCalMinBiasHLT*DCSStatusForSiStripCalMinBias*ALCARECOSiStripCalMinBias)
