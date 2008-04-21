import FWCore.ParameterSet.Config as cms

import copy
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
# AlCaReco for track based alignment using min. bias events
#  module ALCARECOTkAlBeamHaloHLT = hltHighLevel from "HLTrigger/HLTfilters/data/hltHighLevel.cfi"
#  replace ALCARECOTkAlBeamHaloHLT.andOr = true # choose logical OR between Triggerbits
# which is the BeamHalo HLT Tag?
#  replace ALCARECOTkAlBeamHaloHLT.HLTPaths = {""}
ALCARECOTkAlBeamHalo = copy.deepcopy(AlignmentTrackSelector)
seqALCARECOTkAlBeamHalo = cms.Sequence(ALCARECOTkAlBeamHalo)
ALCARECOTkAlBeamHalo.src = 'ctfWithMaterialTracksBeamHaloMuon'
ALCARECOTkAlBeamHalo.filter = True ##do not store empty events

ALCARECOTkAlBeamHalo.applyBasicCuts = True
ALCARECOTkAlBeamHalo.ptMin = 0.0 ##GeV

ALCARECOTkAlBeamHalo.etaMin = -9999
ALCARECOTkAlBeamHalo.etaMax = 9999
ALCARECOTkAlBeamHalo.nHitMin = 3
ALCARECOTkAlBeamHalo.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlBeamHalo.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOTkAlBeamHalo.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlBeamHalo.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlBeamHalo.TwoBodyDecaySelector.applyAcoplanarityFilter = False

