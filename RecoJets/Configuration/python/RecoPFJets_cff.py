import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.sc5PFJets_cfi import sisCone5PFJets
from RecoJets.JetProducers.ic5PFJets_cfi import iterativeCone5PFJets
from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets
from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
from RecoJets.JetProducers.ak5PFJetsTrimmed_cfi import ak5PFJetsTrimmed
from RecoJets.JetProducers.ak5PFJetsFiltered_cfi import ak5PFJetsFiltered, ak5PFJetsMassDropFiltered
from RecoJets.JetProducers.ak5PFJetsPruned_cfi import ak5PFJetsPruned
from CommonTools.ParticleFlow.pfNoPileUpJME_cff  import *
from RecoJets.JetProducers.gk5PFJets_cfi import gk5PFJets
from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
from RecoJets.JetProducers.ca4PFJets_cfi import ca4PFJets
from RecoJets.JetProducers.fixedGridRhoProducer_cfi import fixedGridRhoAll
from RecoJets.JetProducers.fixedGridRhoProducerFastjet_cfi import fixedGridRhoFastjetAll
from RecoJets.JetProducers.caTopTaggers_cff import *
from RecoJets.JetProducers.ak8PFJetsCHS_groomingValueMaps_cfi import ak8PFJetsCHSPrunedLinks, ak8PFJetsCHSFilteredLinks, ak8PFJetsCHSTrimmedLinks
from RecoJets.JetProducers.ca8PFJetsCHS_groomingValueMaps_cfi import ca8PFJetsCHSPrunedLinks, ca8PFJetsCHSFilteredLinks, ca8PFJetsCHSTrimmedLinks

sisCone7PFJets = sisCone5PFJets.clone( rParam = 0.7 )
ak7PFJets = ak5PFJets.clone( rParam = 0.7 )
ak8PFJets = ak5PFJets.clone( rParam = 0.8 )
gk7PFJets = gk5PFJets.clone( rParam = 0.7 )
kt6PFJets = kt4PFJets.clone( rParam = 0.6 )
ca8PFJets = ca4PFJets.clone( rParam = 0.8 )

#compute areas for Fastjet PU subtraction  
kt6PFJets.doRhoFastjet = True
kt6PFJets.doAreaFastjet = True
#use active areas and not Voronoi tessellation for the moment
kt6PFJets.voronoiRfact = 0.9
ak5PFJets.doAreaFastjet = True
ak5PFJetsTrimmed.doAreaFastjet = True
ak7PFJets.doAreaFastjet = True



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


# Advanced Algorithms for AK4, AK5, AK8 and CA8 :
#   - CHS, ungroomed
#   - CHS, pruned
#   - CHS, filtered
#   - CHS, trimmed
ak5PFJetsCHS = ak5PFJets.clone(
    src = cms.InputTag("pfNoPileUpJME")
    )

ak5PFJetsCHSPruned = ak5PFJetsPruned.clone(
    src = cms.InputTag("pfNoPileUpJME")
    )

ak5PFJetsCHSFiltered = ak5PFJetsFiltered.clone(
    src = cms.InputTag("pfNoPileUpJME")
    )

ak5PFJetsCHSTrimmed = ak5PFJetsTrimmed.clone(
    src = cms.InputTag("pfNoPileUpJME")
    )
    
ak4PFJetsCHS = ak5PFJetsCHS.clone(
    rParam = 0.4
    )    

ak8PFJetsCHS = ak5PFJetsCHS.clone(
    rParam = 0.8,
    jetPtMin = 15.0
    )

ak8PFJetsCHSPruned = ak5PFJetsCHSPruned.clone(
    rParam = 0.8,
    jetPtMin = 15.0
    )

ak8PFJetsCHSFiltered = ak5PFJetsCHSFiltered.clone(
    rParam = 0.8,
    jetPtMin = 15.0
    )

ak8PFJetsCHSTrimmed = ak5PFJetsCHSTrimmed.clone(
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
ca15PFJetsCHSMassDropFiltered = ak5PFJetsMassDropFiltered.clone(
    jetAlgorithm = cms.string("CambridgeAachen"),
    src = cms.InputTag("pfNoPileUpJME"),
    rParam = 1.5,
    jetPtMin=100.0
    )

ca15PFJetsCHSFiltered = ak5PFJetsFiltered.clone(
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
                           ak5PFJets+ak8PFJets+
                           pfNoPileUpJMESequence+
                           ak5PFJetsCHS+
                           ak4PFJetsCHS+                           
                           ak8PFJetsCHS+
                           ca8PFJetsCHS+
                           ak8PFJetsCHSConstituents+
                           ca8PFJetsCHSPruned+
                           ca8PFJetsCHSPrunedLinks+
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
                           ak4PFJets+
                           ak5PFJets+ak7PFJets+ak8PFJets+
                           gk5PFJets+gk7PFJets+
                           ca4PFJets+ca8PFJets+
                           pfNoPileUpJMESequence+
                           ak5PFJetsCHS+
                           ak5PFJetsCHSPruned+
                           ak5PFJetsCHSFiltered+
                           ak5PFJetsCHSTrimmed+
                           ak4PFJetsCHS+                                                      
                           ak8PFJetsCHS+
                           ak8PFJetsCHSPruned+
                           ak8PFJetsCHSFiltered+
                           ak8PFJetsCHSTrimmed+
                           ca8PFJetsCHS+
                           ca8PFJetsCHSPruned+
                           ca8PFJetsCHSFiltered+
                           ca8PFJetsCHSTrimmed+
                           ca8PFJetsCHSPrunedLinks+
                           ca8PFJetsCHSTrimmedLinks+
                           ca8PFJetsCHSFilteredLinks+
                           cmsTopTagPFJetsCHS+
                           hepTopTagPFJetsCHS+
                           ca15PFJetsCHSMassDropFiltered+
                           ca15PFJetsCHSFiltered
    )
