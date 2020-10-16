import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.GenJetParticles_cff import genParticlesForJets
from RecoHI.HiJetAlgos.HiGenCleaner_cff import hiPartons
from RecoHI.HiJetAlgos.HiGenJets_cff import ak4HiGenJets
from RecoHI.HiJetAlgos.HiGenCleaner_cff import heavyIonCleanedGenJets

allPartons = cms.EDProducer(
    "PartonSelector",
    src = cms.InputTag('genParticles'),
    withLeptons = cms.bool(False),
    )

cleanedPartons = hiPartons.clone(
    src = 'allPartons',
    )

ak4HiGenJetsCleaned = heavyIonCleanedGenJets.clone(src = "ak4HiGenJets")

cleanedGenJetsTask = cms.Task(
    genParticlesForJets,
    cleanedPartons,
    ak4HiGenJets,
    ak4HiGenJetsCleaned
)

from RecoHI.HiJetAlgos.HiRecoPFJets_cff import PFTowers, pfEmptyCollection, ak4PFJetsForFlow, hiPuRho, hiFJRhoFlowModulation, akCs4PFJets
from RecoHI.HiTracking.highPurityGeneralTracks_cfi import highPurityGeneralTracks
from RecoJets.JetAssociationProducers.ak5JTA_cff import *
from RecoBTag.Configuration.RecoBTag_cff import impactParameterTagInfos, trackCountingHighEffBJetTags, trackCountingHighPurBJetTags, jetProbabilityBJetTags, jetBProbabilityBJetTags, secondaryVertexTagInfos, combinedSecondaryVertexV2BJetTags, simpleSecondaryVertexHighEffBJetTags, simpleSecondaryVertexHighPurBJetTags
ak5JetTracksAssociatorAtVertex
from RecoBTag.SecondaryVertex.simpleSecondaryVertex2TrkComputer_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertex3TrkComputer_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexV2Computer_cfi import *
from RecoBTag.Combined.heavyIonCSVComputer_cfi import *
from RecoBTag.ImpactParameter.jetBProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.jetProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.trackCounting3D2ndComputer_cfi import *
from RecoBTag.ImpactParameter.trackCounting3D3rdComputer_cfi import *
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *

recoPFJetsHIpostAODTask = cms.Task(
    PFTowers,
    pfEmptyCollection,
    ak4PFJetsForFlow,
    hiFJRhoFlowModulation,
    hiPuRho,
    highPurityGeneralTracks,
    akCs4PFJets,
    ak5JetTracksAssociatorAtVertex,
    impactParameterTagInfos,
    trackCountingHighEffBJetTags,
    trackCountingHighPurBJetTags,
    jetProbabilityBJetTags,
    jetBProbabilityBJetTags,
    secondaryVertexTagInfos,
    combinedSecondaryVertexV2BJetTags,
    simpleSecondaryVertexHighEffBJetTags,
    simpleSecondaryVertexHighPurBJetTags,
    patJetCorrFactors
)

recoJetsHIpostAODTask = cms.Task(
    recoPFJetsHIpostAODTask,
    allPartons,
    cleanedGenJetsTask,
    )
