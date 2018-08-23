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

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
#HI-specific algorithms needed in pp scenario special configurations
from RecoHI.HiJetAlgos.hiFJGridEmptyAreaCalculator_cff import hiFJGridEmptyAreaCalculator
pA_2016.toModify(hiFJGridEmptyAreaCalculator, doCentrality = False)

from RecoHI.HiJetAlgos.hiFJRhoProducer import hiFJRhoProducer
from RecoHI.HiJetAlgos.HiRecoPFJets_cff import kt4PFJetsForRho
from RecoHI.HiCentralityAlgos.pACentrality_cfi import pACentrality
pA_2016.toModify(pACentrality, producePixelTracks = False)

_jetHighLevelReco_pA = jetHighLevelReco.copy()
_jetHighLevelReco_pA += kt4PFJetsForRho
_jetHighLevelReco_pA += hiFJRhoProducer
_jetHighLevelReco_pA += hiFJGridEmptyAreaCalculator
_jetHighLevelReco_pA += pACentrality
pA_2016.toReplaceWith(jetHighLevelReco, _jetHighLevelReco_pA)
