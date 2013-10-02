import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.RecoJets_cff import *
from RecoJets.Configuration.JetIDProducers_cff import *
from RecoJets.Configuration.RecoTrackJets_cff import *
from RecoJets.Configuration.RecoJetAssociations_cff import *
from RecoJets.Configuration.RecoPFJets_cff import *
from RecoJets.Configuration.RecoJPTJets_cff import *

jetGlobalReco = cms.Sequence(recoJets*recoJetIds*recoTrackJets)
jetHighLevelReco = cms.Sequence(recoPFJets*recoJetAssociations*recoJetAssociationsExplicit*recoJPTJets)
