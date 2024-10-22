import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4PFJets_cfi import *
from RecoJets.JetProducers.ak8PFJets_cfi import *
from RecoJets.JetProducers.kt4PFJets_cfi import *
from RecoJets.JetProducers.kt6PFJets_cfi import *
from RecoJets.JetProducers.ca15PFJets_cfi import *
from CommonTools.ParticleFlow.pfNoPileUpJME_cff  import *
from CommonTools.PileupAlgos.Puppi_cff import puppi
from CommonTools.PileupAlgos.softKiller_cfi import softKiller
from RecoJets.JetProducers.fixedGridRhoProducer_cfi import fixedGridRhoAll
from RecoJets.JetProducers.fixedGridRhoProducerFastjet_cfi import fixedGridRhoFastjetAll
from RecoJets.JetProducers.ak8PFJetsPuppi_groomingValueMaps_cfi import ak8PFJetsPuppiSoftDropMass


fixedGridRhoFastjetCentral = fixedGridRhoFastjetAll.clone(
    maxRapidity = 2.5
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
                           pfNoPileUpJMETask,
                           ak4PFJetsCHS,
                           puppi,
                           ak4PFJetsPuppi,
                           ak8PFJetsPuppi,
                           ak8PFJetsPuppiConstituents,
                           ak8PFJetsPuppiSoftDrop,
                           ak8PFJetsPuppiSoftDropMass
    )
recoPFJets   = cms.Sequence(recoPFJetsTask)

recoAllPFJetsTask=cms.Task(fixedGridRhoAll,
                           fixedGridRhoFastjetAll,
                           fixedGridRhoFastjetCentral,
                           fixedGridRhoFastjetCentralChargedPileUp,
                           fixedGridRhoFastjetCentralNeutral,
                           ak4PFJets,
                           ak8PFJets,
                           pfNoPileUpJMETask,
                           ak4PFJetsCHS,
                           puppi,
                           ak4PFJetsPuppi,
                           ak8PFJetsPuppi,
                           ak8PFJetsPuppiSoftDrop,
                           ak8PFJetsPuppiSoftDropMass
    )
recoAllPFJets=cms.Sequence(recoAllPFJetsTask)

recoPFJetsWithSubstructureTask=cms.Task(
                           fixedGridRhoAll,
                           fixedGridRhoFastjetAll,
                           fixedGridRhoFastjetCentral,
                           fixedGridRhoFastjetCentralChargedPileUp,
                           fixedGridRhoFastjetCentralNeutral,
                           ak4PFJets,
                           ak8PFJets,
                           pfNoPileUpJMETask,
                           ak4PFJetsCHS,
                           puppi,
                           ak4PFJetsPuppi,
                           ak8PFJetsPuppi,
                           ak8PFJetsPuppiSoftDrop,
                           ak8PFJetsPuppiConstituents,
                           ak8PFJetsPuppiSoftDropMass,
                           ak8PFJetsCS,
                           ak8PFJetsCSConstituents,                           
                           softKiller,
                           ak4PFJetsSK
    )
recoPFJetsWithSubstructure=cms.Sequence(recoPFJetsWithSubstructureTask)

from RecoHI.HiJetAlgos.HiRecoPFJets_cff import PFTowers, akPu3PFJets, akPu4PFJets, kt4PFJetsForRho, ak4PFJetsForFlow, akCs4PFJets, pfEmptyCollection, hiFJRhoFlowModulation, hiPuRho, hiPFCandCleanerforJets
from RecoHI.HiJetAlgos.hiFJRhoProducer import hiFJRhoProducer


recoPFJetsHITask =cms.Task(fixedGridRhoAll,
                           fixedGridRhoFastjetAll,
                           fixedGridRhoFastjetCentral,
                           fixedGridRhoFastjetCentralChargedPileUp,
                           fixedGridRhoFastjetCentralNeutral,
                           pfEmptyCollection,
                           ak4PFJets,
                           ak4PFJetsCHS,
                           PFTowers,			   
                           akPu3PFJets,
                           akPu4PFJets,
                           kt4PFJetsForRho,
                           hiPFCandCleanerforJets,
                           ak4PFJetsForFlow,
                           hiFJRhoProducer,
                           hiPuRho,
                           hiFJRhoFlowModulation,
                           akCs4PFJets
   )
recoPFJetsHI   = cms.Sequence(recoPFJetsHITask)
 
