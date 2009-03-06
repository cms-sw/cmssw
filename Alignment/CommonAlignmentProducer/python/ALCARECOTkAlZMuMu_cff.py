import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# AlCaReco for track based alignment using ZMuMu events
ALCARECOTkAlZMuMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlZMuMu',
    throw = False # tolerate triggers stated above, but not available
    )

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlZMuMu = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlZMuMu.filter = True ##do not store empty events

ALCARECOTkAlZMuMu.applyBasicCuts = True
ALCARECOTkAlZMuMu.ptMin = 15.0 ##GeV

ALCARECOTkAlZMuMu.etaMin = -3.5
ALCARECOTkAlZMuMu.etaMax = 3.5
ALCARECOTkAlZMuMu.nHitMin = 0
ALCARECOTkAlZMuMu.GlobalSelector.applyIsolationtest = True
ALCARECOTkAlZMuMu.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.minXMass = 70.0 ##GeV

ALCARECOTkAlZMuMu.TwoBodyDecaySelector.maxXMass = 110.0 ##GeV

ALCARECOTkAlZMuMu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)

ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyChargeFilter = True
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.charge = 0
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlZMuMu = cms.Sequence(ALCARECOTkAlZMuMuHLT+ALCARECOTkAlZMuMu)
