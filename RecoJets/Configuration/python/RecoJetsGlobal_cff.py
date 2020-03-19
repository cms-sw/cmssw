import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.RecoJets_cff import *
from RecoJets.Configuration.JetIDProducers_cff import *
from RecoJets.Configuration.RecoTrackJets_cff import *
from RecoJets.Configuration.RecoJetAssociations_cff import *
from RecoJets.Configuration.RecoPFJets_cff import *
from RecoJets.Configuration.RecoJPTJets_cff import *
from JetMETCorrections.Configuration.JetCorrectorsForReco_cff import *

jetGlobalRecoTask = cms.Task(recoJetsTask, 
                             recoJetIdsTask, 
                             recoTrackJetsTask)
jetGlobalReco = cms.Sequence(jetGlobalRecoTask)
jetHighLevelRecoTask = cms.Task(recoPFJetsTask,
                                jetCorrectorsForRecoTask,
                                recoJetAssociationsTask,
                                recoJetAssociationsExplicitTask,
                                recoJPTJetsTask)
jetHighLevelReco = cms.Sequence(jetHighLevelRecoTask)

from RecoHI.HiJetAlgos.hiFJGridEmptyAreaCalculator_cff import hiFJGridEmptyAreaCalculator
from RecoHI.HiJetAlgos.hiFJRhoProducer import hiFJRhoProducer
from RecoHI.HiJetAlgos.HiRecoPFJets_cff import kt4PFJetsForRho
from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from RecoHI.HiCentralityAlgos.pACentrality_cfi import pACentrality
pA_2016.toModify(pACentrality, producePixelTracks = False)

_jetHighLevelReco_pATask = jetHighLevelRecoTask.copy()
_jetHighLevelReco_pATask.add(kt4PFJetsForRho)
_jetHighLevelReco_pATask.add(hiFJRhoProducer)
_jetHighLevelReco_pATask.add(hiFJGridEmptyAreaCalculator)
_jetHighLevelReco_pATask.add(pACentrality)
pA_2016.toReplaceWith(jetHighLevelRecoTask, _jetHighLevelReco_pATask)

_jetGlobalReco_HITask = cms.Task(recoJetsHITask,recoJetIdsTask)
_jetGlobalReco_HI = cms.Sequence(_jetGlobalReco_HITask)
_jetHighLevelReco_HITask = cms.Task(recoPFJetsHITask,jetCorrectorsForRecoTask,recoJetAssociationsTask)
_jetHighLevelReco_HI = cms.Sequence(_jetHighLevelReco_HITask)

pp_on_AA_2018.toReplaceWith(jetGlobalRecoTask,_jetGlobalReco_HITask)
pp_on_AA_2018.toReplaceWith(jetHighLevelRecoTask,_jetHighLevelReco_HITask)
