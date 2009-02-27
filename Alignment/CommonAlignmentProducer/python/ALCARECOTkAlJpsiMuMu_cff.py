# AlCaReco for track based alignment using J/Psi->MuMu events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlJpsiMuMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    HLTPaths = ['HLT_DoubleMu3'],
    throw = False # tolerate triggers stated above, but not available
    )

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlJpsiMuMu = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlJpsiMuMu.filter = True ##do not store empty events

ALCARECOTkAlJpsiMuMu.applyBasicCuts = True
ALCARECOTkAlJpsiMuMu.ptMin = 0.8 ##GeV

ALCARECOTkAlJpsiMuMu.etaMin = -3.5
ALCARECOTkAlJpsiMuMu.etaMax = 3.5
ALCARECOTkAlJpsiMuMu.nHitMin = 0
ALCARECOTkAlJpsiMuMu.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlJpsiMuMu.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.minXMass = 3.0 ##GeV

ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.maxXMass = 3.2 ##GeV

ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)

ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.charge = 0
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOTkAlJpsiMuMu.TwoBodyDecaySelector.acoplanarDistance = 1 ##radian


seqALCARECOTkAlJpsiMuMu = cms.Sequence(ALCARECOTkAlJpsiMuMuHLT+ALCARECOTkAlJpsiMuMu)
