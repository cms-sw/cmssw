# AlCaReco for track based alignment using Upsilon->MuMu events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlUpsilonMuMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    HLTPaths = ['HLT DoubleMu3'],
    throw = False # tolerate triggers stated above, but not available
    )

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlUpsilonMuMu = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlUpsilonMuMu.filter = True ##do not store empty events

ALCARECOTkAlUpsilonMuMu.applyBasicCuts = True
ALCARECOTkAlUpsilonMuMu.ptMin = 0.8 ##GeV

ALCARECOTkAlUpsilonMuMu.etaMin = -3.5
ALCARECOTkAlUpsilonMuMu.etaMax = 3.5
ALCARECOTkAlUpsilonMuMu.nHitMin = 0
ALCARECOTkAlUpsilonMuMu.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlUpsilonMuMu.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.minXMass = 9.25 ##GeV

ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.maxXMass = 9.8 ##GeV

ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)

ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.applyChargeFilter = True
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.charge = 0
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.acoplanarDistance = 1 ##radian

seqALCARECOTkAlUpsilonMuMu = cms.Sequence(ALCARECOTkAlUpsilonMuMuHLT+ALCARECOTkAlUpsilonMuMu)

