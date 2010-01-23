# AlCaReco for track based alignment using min. bias events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
#Note the MinBias selection should contain as many tracks as possible but no overlaps. So the HLT selection selects any event that is not selected in another TkAl* selector.
ALCARECOTkAlMinBiasNOTHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlMinBiasNOT',
    throw = False # tolerate triggers stated above, but not available
    )
ALCARECOTkAlMinBiasHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlMinBias',
    throw = False # tolerate triggers stated above, but not available
    )



import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlMinBias = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlMinBias.filter = True ##do not store empty events	

ALCARECOTkAlMinBias.applyBasicCuts = True
ALCARECOTkAlMinBias.pMin = 1.5 ##GeV


ALCARECOTkAlMinBias.etaMin = -3.5
ALCARECOTkAlMinBias.etaMax = 3.5
ALCARECOTkAlMinBias.nHitMin = 5 ## at least 5 hits required
ALCARECOTkAlMinBias.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlMinBias.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlMinBias = cms.Sequence(ALCARECOTkAlMinBiasHLT*
~ALCARECOTkAlMinBiasNOTHLT+ALCARECOTkAlMinBias)
