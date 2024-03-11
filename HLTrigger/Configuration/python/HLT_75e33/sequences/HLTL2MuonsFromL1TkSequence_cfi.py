import FWCore.ParameterSet.Config as cms

from ..modules.hltL2MuonSeedsFromL1TkMuon_cfi import *
from ..modules.hltL2MuonsFromL1TkMuon_cfi import *
from ..modules.hltL2OfflineMuonSeeds_cfi import *
from ..sequences.muonlocalrecoSequence_cfi import *

HLTL2MuonsFromL1TkSequence = cms.Sequence(muonlocalrecoSequence+hltL2OfflineMuonSeeds+hltL2MuonSeedsFromL1TkMuon+hltL2MuonsFromL1TkMuon)
