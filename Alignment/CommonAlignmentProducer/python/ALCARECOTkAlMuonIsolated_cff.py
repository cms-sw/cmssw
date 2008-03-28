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

ALCARECOTkAlMuonIsolatedHLT.HLTPaths = ['HLT1MuonIso']
ALCARECOTkAlMuonIsolated.applyBasicCuts = True
ALCARECOTkAlMuonIsolated.ptMin = 11.0 ##GeV

ALCARECOTkAlMuonIsolated.etaMin = -3.5
ALCARECOTkAlMuonIsolated.etaMax = 3.5
ALCARECOTkAlMuonIsolated.nHitMin = 0
ALCARECOTkAlMuonIsolated.GlobalSelector.applyIsolationtest = True
ALCARECOTkAlMuonIsolated.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyAcoplanarityFilter = False

