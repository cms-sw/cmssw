import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for track based alignment using isolated muon tracks
ALCARECOTkAlMuonIsolatedHLT = copy.deepcopy(hltHighLevel)
import copy
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
ALCARECOTkAlMuonIsolated = copy.deepcopy(AlignmentTrackSelector)
seqALCARECOTkAlMuonIsolated = cms.Sequence(ALCARECOTkAlMuonIsolatedHLT+ALCARECOTkAlMuonIsolated)
ALCARECOTkAlMuonIsolatedHLT.andOr = True ## choose logical OR between Triggerbits

# NOTE: this has to hold for all triggertables, so in order for this to not crash all bits of all triggertables 
#         have to be in all triggertables but switched off via prescale.
#for L = 10e30:  HLT1MuonPrescalePt3 , HLT1MuonPrescalePt5
#for L = 10e31:  unknown
#for L = 10e32:  HLT1MuonIso , HLT1MuonNonIso
ALCARECOTkAlMuonIsolatedHLT.HLTPaths = ['HLT1MuonPrescalePt3', 'HLT1MuonPrescalePt5', 'HLT1MuonIso', 'HLT1MuonNonIso']
ALCARECOTkAlMuonIsolated.filter = True ##do not store empty events

ALCARECOTkAlMuonIsolated.applyBasicCuts = True
ALCARECOTkAlMuonIsolated.ptMin = 2.0 ##GeV 

ALCARECOTkAlMuonIsolated.etaMin = -3.5
ALCARECOTkAlMuonIsolated.etaMax = 3.5
ALCARECOTkAlMuonIsolated.nHitMin = 0
ALCARECOTkAlMuonIsolated.GlobalSelector.applyIsolationtest = True
ALCARECOTkAlMuonIsolated.GlobalSelector.minJetDeltaR = 0.1
ALCARECOTkAlMuonIsolated.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyAcoplanarityFilter = False

