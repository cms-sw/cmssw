import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# AlCaReco for track based alignment using WMuNu events
ALCARECOTkAlWMuNuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlWMuNu = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
seqALCARECOTkAlWMuNu = cms.Sequence(ALCARECOTkAlWMuNuHLT+ALCARECOTkAlWMuNu)
ALCARECOTkAlWMuNuHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlWMuNuHLT.HLTPaths = ['HLT1MuonIso'] ##these need further studies

ALCARECOTkAlWMuNu.filter = True ##do not store empty events

ALCARECOTkAlWMuNu.applyBasicCuts = True
ALCARECOTkAlWMuNu.ptMin = 20.0 ##GeV

ALCARECOTkAlWMuNu.etaMin = -3.5
ALCARECOTkAlWMuNu.etaMax = 3.5
ALCARECOTkAlWMuNu.nHitMin = 0
ALCARECOTkAlWMuNu.GlobalSelector.applyIsolationtest = True
ALCARECOTkAlWMuNu.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlWMuNu.GlobalSelector.applyJetCountFilter = True
ALCARECOTkAlWMuNu.GlobalSelector.minJetPt = 40 ##GeV

ALCARECOTkAlWMuNu.GlobalSelector.maxJetCount = 3
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.applyMissingETFilter = True
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.minXMass = 40.0 ##GeV

ALCARECOTkAlWMuNu.TwoBodyDecaySelector.maxXMass = 200.0 ##GeV

ALCARECOTkAlWMuNu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)

ALCARECOTkAlWMuNu.TwoBodyDecaySelector.applyChargeFilter = True
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.charge = 1
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.useUnsignedCharge = True
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.applyAcoplanarityFilter = True
ALCARECOTkAlWMuNu.TwoBodyDecaySelector.acoplanarDistance = 1 ##radian


