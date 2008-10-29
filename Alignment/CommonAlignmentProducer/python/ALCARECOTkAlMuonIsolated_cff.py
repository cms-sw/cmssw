# AlCaReco for track based alignment using isolated muon tracks
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlMuonIsolatedHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    # NOTE: this has to hold for all triggertables, so in order for this to not crash all bits of all triggertables 
    #         have to be in all triggertables but switched off via prescale.
    #for L = 10e30:  HLT_Mu3 , HLT_Mu5
    #for L = 10e31:  unknown
    #for L = 10e32:  HLT_IsoMu11 , HLT_Mu15_L1Mu7
    HLTPaths = ['HLT_Mu3', 'HLT_Mu5', 'HLT_IsoMu11', 'HLT_Mu15'],
    throw = False # tolerate triggers stated above, but not available
    )

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlMuonIsolated = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    filter = True, ##do not store empty events
    applyBasicCuts = True,
    ptMin = 2.0, ##GeV 
    etaMin = -3.5,
    etaMax = 3.5,
    nHitMin = 0
    )
# These unfortunately cannot be put into the clone(..): 
ALCARECOTkAlMuonIsolated.GlobalSelector.applyIsolationtest = True
ALCARECOTkAlMuonIsolated.GlobalSelector.minJetDeltaR = 0.1
ALCARECOTkAlMuonIsolated.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlMuonIsolated = cms.Sequence(ALCARECOTkAlMuonIsolatedHLT+ALCARECOTkAlMuonIsolated)
