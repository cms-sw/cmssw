import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4PFJets_cfi import *
from RecoJets.JetProducers.ak8PFJets_cfi import *
from RecoJets.JetProducers.kt4PFJets_cfi import *
from RecoJets.JetProducers.kt6PFJets_cfi import *
from RecoJets.JetProducers.ca15PFJets_cfi import *
from RecoJets.JetProducers.caTopTaggers_cff import cmsTopTagPFJetsCHS
from CommonTools.ParticleFlow.pfNoPileUpJME_cff  import *
from CommonTools.PileupAlgos.Puppi_cff import puppi
from CommonTools.PileupAlgos.softKiller_cfi import softKiller
from RecoJets.JetProducers.fixedGridRhoProducer_cfi import fixedGridRhoAll
from RecoJets.JetProducers.fixedGridRhoProducerFastjet_cfi import fixedGridRhoFastjetAll
from RecoJets.JetProducers.ak8PFJetsCHS_groomingValueMaps_cfi import ak8PFJetsCHSPrunedMass, ak8PFJetsCHSFilteredMass, ak8PFJetsCHSTrimmedMass, ak8PFJetsCHSSoftDropMass

fixedGridRhoFastjetCentral = fixedGridRhoFastjetAll.clone(
    maxRapidity = cms.double(2.5)
    )

fixedGridRhoFastjetCentralChargedPileUp = fixedGridRhoFastjetAll.clone(
    pfCandidatesTag = "pfPileUpAllChargedParticles",
    maxRapidity = 2.5
    )

fixedGridRhoFastjetCentralNeutral = fixedGridRhoFastjetAll.clone(
    pfCandidatesTag = "pfAllNeutralHadronsAndPhotons",
    maxRapidity = 2.5
    )

recoPFJetsTask   =cms.Task(fixedGridRhoAll,
                           fixedGridRhoFastjetAll,
                           fixedGridRhoFastjetCentral,
                           fixedGridRhoFastjetCentralChargedPileUp,
                           fixedGridRhoFastjetCentralNeutral,
                           ak4PFJets,
                           ak4PFJetsCHS,
                           ak8PFJetsCHS,
                           ak8PFJetsCHSConstituents,
                           ak8PFJetsCHSSoftDrop,
                           ak8PFJetsCHSSoftDropMass,
                           cmsTopTagPFJetsCHS,
                           pfNoPileUpJMETask
    )
recoPFJets   = cms.Sequence(recoPFJetsTask)

recoAllPFJetsTask=cms.Task(fixedGridRhoAll,
                           fixedGridRhoFastjetAll,
                           fixedGridRhoFastjetCentral,
                           fixedGridRhoFastjetCentralChargedPileUp,
                           fixedGridRhoFastjetCentralNeutral,
                           ak4PFJets,ak8PFJets,
                           pfNoPileUpJMETask,
                           ak8PFJetsCHS,
                           ak8PFJetsCHSPruned,
                           ak8PFJetsCHSFiltered,
                           ak8PFJetsCHSTrimmed,
                           ak8PFJetsCHSSoftDrop,
                           ak4PFJetsCHS, 
                           ak8PFJetsCHS,
                           ak8PFJetsCHSPruned,
                           ak8PFJetsCHSFiltered,
                           ak8PFJetsCHSTrimmed,
                           ak8PFJetsCHSSoftDrop,
                           ak8PFJetsCHSPrunedMass,
                           ak8PFJetsCHSTrimmedMass,
                           ak8PFJetsCHSSoftDropMass,
                           ak8PFJetsCHSFilteredMass,
                           ca15PFJetsCHSMassDropFiltered,
                           ca15PFJetsCHSFiltered
    )
recoAllPFJets=cms.Sequence(recoAllPFJetsTask)

recoPFJetsWithSubstructureTask=cms.Task(
                           fixedGridRhoAll,
                           fixedGridRhoFastjetAll,
                           fixedGridRhoFastjetCentral,
                           fixedGridRhoFastjetCentralChargedPileUp,
                           fixedGridRhoFastjetCentralNeutral,
                           ak4PFJets,ak8PFJets,
                           pfNoPileUpJMETask,
                           ak8PFJetsCHS,
                           ak8PFJetsCHSPruned,
                           ak8PFJetsCHSFiltered,
                           ak8PFJetsCHSTrimmed,
                           ak8PFJetsCHSSoftDrop,
                           ak4PFJetsCHS,                                                      
                           ak8PFJetsCHS,
                           ak8PFJetsCHSPruned,
                           ak8PFJetsCHSFiltered,
                           ak8PFJetsCHSTrimmed,
                           ak8PFJetsCHSSoftDrop,
                           ak8PFJetsCHSConstituents,
                           ak8PFJetsCHSPrunedMass,
                           ak8PFJetsCHSTrimmedMass,
                           ak8PFJetsCHSSoftDropMass,
                           ak8PFJetsCHSFilteredMass,
                           ca15PFJetsCHSMassDropFiltered,
                           ca15PFJetsCHSFiltered,
                           ak8PFJetsCS,
                           ak8PFJetsCSConstituents,
                           puppi,
                           ak4PFJetsPuppi,
                           softKiller,
                           ak4PFJetsSK
    )
recoPFJetsWithSubstructure=cms.Sequence(recoPFJetsWithSubstructureTask)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018

from RecoJets.JetProducers.HiRecoPFJets_cff import PFTowers, akPu3PFJets, akPu4PFJets, akPu5PFJets, kt4PFJetsForRhoHI, hiFJRhoProducer, akCs3PFJets, akCs4PFJets

_recoPFJetsHI = recoPFJets.copy()
_recoPFJetsHI += PFTowers
_recoPFJetsHI += akPu3PFJets
_recoPFJetsHI += akPu4PFJets
_recoPFJetsHI += akPu5PFJets
_recoPFJetsHI += kt4PFJetsForRhoHI
_recoPFJetsHI += hiFJRhoProducer
_recoPFJetsHI += akCs3PFJets
_recoPFJetsHI += akCs4PFJets

for e in [pp_on_AA_2018]:
 e.toReplaceWith(recoPFJets, _recoPFJetsHI)
