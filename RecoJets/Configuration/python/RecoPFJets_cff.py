import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.sc5PFJets_cfi import sisCone5PFJets
from RecoJets.JetProducers.ic5PFJets_cfi import iterativeCone5PFJets
from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets
from RecoJets.JetProducers.ak4PFJetsTrimmed_cfi import ak4PFJetsTrimmed
from RecoJets.JetProducers.ak4PFJetsFiltered_cfi import ak4PFJetsFiltered, ak4PFJetsMassDropFiltered
from RecoJets.JetProducers.ak4PFJetsPruned_cfi import ak4PFJetsPruned
from CommonTools.ParticleFlow.pfNoPileUpJME_cff  import *
from RecoJets.JetProducers.gk5PFJets_cfi import gk5PFJets
from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
from RecoJets.JetProducers.ca4PFJets_cfi import ca4PFJets
from RecoJets.JetProducers.fixedGridRhoProducer_cfi import fixedGridRhoAll
from RecoJets.JetProducers.fixedGridRhoProducerFastjet_cfi import fixedGridRhoFastjetAll
from RecoJets.JetProducers.caTopTaggers_cff import *

sisCone7PFJets = sisCone5PFJets.clone( rParam = 0.7 )
ak7PFJets = ak4PFJets.clone( rParam = 0.7 )
ak8PFJets = ak4PFJets.clone( rParam = 0.8 )
gk7PFJets = gk5PFJets.clone( rParam = 0.7 )
kt6PFJets = kt4PFJets.clone( rParam = 0.6 )
ca8PFJets = ca4PFJets.clone( rParam = 0.8 )

#compute areas for Fastjet PU subtraction  
kt6PFJets.doRhoFastjet = True
kt6PFJets.doAreaFastjet = True
#use active areas and not Voronoi tessellation for the moment
kt6PFJets.voronoiRfact = 0.9
ak4PFJets.doAreaFastjet = True
ak5PFJets.doAreaFastjet = True
ak4PFJetsTrimmed.doAreaFastjet = True
ak4PFJetsPruned.doAreaFastjet = True
ak4PFJetsFiltered.doAreaFastjet = True
ak8PFJets.doAreaFastjet = True



kt6PFJetsCentralChargedPileUp = kt6PFJets.clone(
    src = cms.InputTag("pfPileUpAllChargedParticles"),
    Ghost_EtaMax = cms.double(3.1),
    Rho_EtaMax = cms.double(2.5)
    )

kt6PFJetsCentralNeutral = kt6PFJets.clone(
    src = cms.InputTag("pfAllNeutralHadronsAndPhotons"),
    Ghost_EtaMax = cms.double(3.1),
    Rho_EtaMax = cms.double(2.5),
    inputEtMin = cms.double(0.5)
    )



kt6PFJetsCentralNeutralTight = kt6PFJetsCentralNeutral.clone(
    inputEtMin = cms.double(1.0)
    )



fixedGridRhoFastjetCentralChargedPileUp = fixedGridRhoFastjetAll.clone(
    src = cms.InputTag("pfPileUpAllChargedParticles"),
    maxRapidity = cms.double(2.5)
    )

fixedGridRhoFastjetCentralNeutral = fixedGridRhoFastjetAll.clone(
    src = cms.InputTag("pfAllNeutralHadronsAndPhotons"),
    maxRapidity = cms.double(2.5)
    )



ak8PFJetsCHSConstituents = cms.EDFilter("PFJetConstituentSelector",
                                        src = cms.InputTag("ak8PFJetsCHS"),
                                        cut = cms.string("pt > 100.0 && abs(rapidity()) < 2.4")
                                        )


# Advanced Algorithms for AK4, AK8 and CA8 :
#   - CHS, ungroomed
#   - CHS, pruned
#   - CHS, filtered
#   - CHS, trimmed
ak4PFJetsCHS = ak4PFJets.clone(
    src = cms.InputTag("pfNoPileUpJME")
    )

ak4PFJetsCHSPruned = ak4PFJetsPruned.clone(
    src = cms.InputTag("pfNoPileUpJME")
    )

ak4PFJetsCHSFiltered = ak4PFJetsFiltered.clone(
    src = cms.InputTag("pfNoPileUpJME")
    )

ak4PFJetsCHSTrimmed = ak4PFJetsTrimmed.clone(
    src = cms.InputTag("pfNoPileUpJME")
    )

ak8PFJetsCHS = ak4PFJetsCHS.clone(
    rParam = 0.8,
    jetPtMin = 15.0
    )

ak8PFJetsCHSPruned = ak4PFJetsCHSPruned.clone(
    rParam = 0.8,
    jetPtMin = 15.0
    )

ak8PFJetsCHSFiltered = ak4PFJetsCHSFiltered.clone(
    rParam = 0.8,
    jetPtMin = 15.0
    )

ak8PFJetsCHSTrimmed = ak4PFJetsCHSTrimmed.clone(
    rParam = 0.8,
    jetPtMin = 15.0
    )

ca8PFJetsCHS = ak8PFJetsCHS.clone(
    jetAlgorithm = cms.string("CambridgeAachen")
    )

ca8PFJetsCHSPruned = ak8PFJetsCHSPruned.clone(
    jetAlgorithm = cms.string("CambridgeAachen")
    )

ca8PFJetsCHSFiltered = ak8PFJetsCHSFiltered.clone(
    jetAlgorithm = cms.string("CambridgeAachen")
    )

ca8PFJetsCHSTrimmed = ak8PFJetsCHSTrimmed.clone(
    jetAlgorithm = cms.string("CambridgeAachen")
    )


# Higgs taggers
ca15PFJetsCHSMassDropFiltered = ak4PFJetsMassDropFiltered.clone(
    jetAlgorithm = cms.string("CambridgeAachen"),
    src = cms.InputTag("pfNoPileUpJME"),
    rParam = 1.5,
    jetPtMin=100.0
    )

ca15PFJetsCHSFiltered = ak4PFJetsFiltered.clone(
    jetAlgorithm = cms.string("CambridgeAachen"),
    src = cms.InputTag("pfNoPileUpJME"),
    rParam = 1.5,
    jetPtMin=100.0
    )

cmsTopTagPFJetsCHS.src = cms.InputTag("ak8PFJetsCHSConstituents", "constituents")
hepTopTagPFJetsCHS.src = cms.InputTag("ak8PFJetsCHSConstituents", "constituents")

recoPFJets   =cms.Sequence(#kt4PFJets+kt6PFJets+
                           iterativeCone5PFJets+
                           #kt6PFJetsCentralChargedPileUp+
                           #kt6PFJetsCentralNeutral+
                           #kt6PFJetsCentralNeutralTight+
                           fixedGridRhoAll+
                           fixedGridRhoFastjetAll+
                           fixedGridRhoFastjetCentralChargedPileUp+
                           fixedGridRhoFastjetCentralNeutral+
                           ak4PFJets+
                           # TO BE ADDED BACK AFTER CHECKS : 
                           #ak5PFJets+
                           ak8PFJets+
                           pfNoPileUpJMESequence+
                           ak4PFJetsCHS+
                           ak8PFJetsCHS+
                           ca8PFJetsCHS+
                           ak8PFJetsCHSConstituents+
                           ca8PFJetsCHSPruned+
                           cmsTopTagPFJetsCHS+
                           hepTopTagPFJetsCHS+
                           ca15PFJetsCHSMassDropFiltered+
                           ca15PFJetsCHSFiltered
    )

recoAllPFJets=cms.Sequence(sisCone5PFJets+sisCone7PFJets+
                           iterativeCone5PFJets+
                           kt4PFJets+kt6PFJets+
                           kt6PFJetsCentralChargedPileUp+
                           kt6PFJetsCentralNeutral+
                           kt6PFJetsCentralNeutralTight+
                           fixedGridRhoAll+
                           fixedGridRhoFastjetAll+
                           fixedGridRhoFastjetCentralChargedPileUp+
                           fixedGridRhoFastjetCentralNeutral+
                           iterativeCone5PFJets+
                           ak4PFJets+ak5PFJets+ak8PFJets+
                           gk5PFJets+gk7PFJets+
                           ca4PFJets+ca8PFJets+
                           pfNoPileUpJMESequence+
                           ak4PFJetsCHS+
                           ak4PFJetsCHSPruned+
                           ak4PFJetsCHSFiltered+
                           ak4PFJetsCHSTrimmed+
                           ak8PFJetsCHS+
                           ak8PFJetsCHSPruned+
                           ak8PFJetsCHSFiltered+
                           ak8PFJetsCHSTrimmed+
                           ca8PFJetsCHS+
                           ca8PFJetsCHSPruned+
                           ca8PFJetsCHSFiltered+
                           ca8PFJetsCHSTrimmed+
                           cmsTopTagPFJetsCHS+
                           hepTopTagPFJetsCHS+
                           ca15PFJetsCHSMassDropFiltered+
                           ca15PFJetsCHSFiltered
    )
