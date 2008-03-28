import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.RecoJets_cff import *
from RecoMET.Configuration.RecoMET_cff import *
caloJetMet = cms.Sequence(recoJets)

