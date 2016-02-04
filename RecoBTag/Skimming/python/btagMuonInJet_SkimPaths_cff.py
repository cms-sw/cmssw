import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagMuonInJet_HLT_cfi import *
from RecoBTag.Skimming.btagMuonInJet_cfi import *
btagMuonInJetPath = cms.Path(btagMuonInJet_HLT*btagMuonInJet)

