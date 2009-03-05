# AlCaReco for track based alignment using min. bias events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
#Note the MinBias selection should contain as many tracks as possible but no overlaps. So the HLT selection selects any event that is not selected in another TkAl* selector.
ALCARECOTkAlMinBiasNOTHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    HLTPaths = [
        #TkAlMuonIsolated
        "HLT_L1MuOpen", "HLT_L1Mu", "HLT_L2Mu9",
        "HLT_Mu3", "HLT_Mu5", "HLT_Mu9", "HLT_Mu11",
        #TkAlMuonIsolated _and_ TkAl[Z,Jpsi,Upsilon]MuMu
        "HLT_DoubleMu3",
        #TkAlCosmics{0T}HLT
        'HLT_TrackerCosmics'
        ],
    throw = False # tolerate triggers stated above, but not available
    )


import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlMinBias = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlMinBias.filter = True ##do not store empty events	

ALCARECOTkAlMinBias.applyBasicCuts = True
ALCARECOTkAlMinBias.ptMin = 1.5 ##GeV

ALCARECOTkAlMinBias.etaMin = -3.5
ALCARECOTkAlMinBias.etaMax = 3.5
ALCARECOTkAlMinBias.nHitMin = 0
ALCARECOTkAlMinBias.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlMinBias.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlMinBias = cms.Sequence(~ALCARECOTkAlMinBiasNOTHLT+ALCARECOTkAlMinBias)
