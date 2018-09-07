import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.RecoJets_cff import *
from RecoJets.Configuration.JetIDProducers_cff import *
from RecoJets.Configuration.RecoTrackJets_cff import *
from RecoJets.Configuration.RecoJetAssociations_cff import *
from RecoJets.Configuration.RecoPFJets_cff import *
from RecoJets.Configuration.RecoJPTJets_cff import *
from JetMETCorrections.Configuration.JetCorrectorsForReco_cff import *

jetGlobalReco = cms.Sequence(recoJets*recoJetIds*recoTrackJets)
jetHighLevelReco = cms.Sequence(recoPFJets*jetCorrectorsForReco*recoJetAssociations*recoJetAssociationsExplicit*recoJPTJets)

from RecoHI.HiJetAlgos.hiFJGridEmptyAreaCalculator_cff import hiFJGridEmptyAreaCalculator
from RecoHI.HiJetAlgos.hiFJRhoProducer import hiFJRhoProducer
from RecoHI.HiJetAlgos.HiRecoPFJets_cff import kt4PFJetsForRho
from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from RecoHI.HiCentralityAlgos.pACentrality_cfi import pACentrality
pA_2016.toModify(pACentrality, producePixelTracks = False)

_jetHighLevelReco_pA = jetHighLevelReco.copy()
_jetHighLevelReco_pA += kt4PFJetsForRho
_jetHighLevelReco_pA += hiFJRhoProducer
_jetHighLevelReco_pA += hiFJGridEmptyAreaCalculator
_jetHighLevelReco_pA += pACentrality
pA_2016.toReplaceWith(jetHighLevelReco, _jetHighLevelReco_pA)

_jetGlobalReco_HI = cms.Sequence(recoJetsHI*recoJetIds)
_jetHighLevelReco_HI = cms.Sequence(recoPFJetsHI*jetCorrectorsForReco*recoJetAssociations)

pp_on_AA_2018.toReplaceWith(jetGlobalReco,_jetGlobalReco_HI)
pp_on_AA_2018.toReplaceWith(jetHighLevelReco,_jetHighLevelReco_HI)
