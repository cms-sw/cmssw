import FWCore.ParameterSet.Config as cms

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

seqALCARECOSiStripCalMinBias = cms.Sequence(ALCARECOSiStripCalMinBiasHLT*ALCARECOSiStripCalMinBias)
