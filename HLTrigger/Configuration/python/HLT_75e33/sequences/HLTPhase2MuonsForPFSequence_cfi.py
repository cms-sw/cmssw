import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2L3MuonLinks_cfi import *
from ..modules.hltPhase2MuonLinksForPF_cfi import *
from ..modules.hltPhase2MuonsForPF_cfi import *

#Sequence to have muons inner track pointing to hltGeneralTracks
HLTPhase2MuonsForPFSequence = cms.Sequence(
    hltPhase2L3MuonLinks
    + hltPhase2MuonLinksForPF
    + hltPhase2MuonsForPF
)
