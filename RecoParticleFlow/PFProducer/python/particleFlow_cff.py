import FWCore.ParameterSet.Config as cms

#Geometry
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cfi import *
# jet reconstruction --------------------------------------------------
from RecoJets.Configuration.RecoJets_cff import *
from RecoParticleFlow.PFProducer.particleFlowJetCandidates_cfi import *
iterativeCone5PFJets = cms.EDProducer("IterativeConeJetProducer",
    IconeJetParameters,
    src = cms.InputTag("particleFlowJetCandidates"),
    inputEtMin = cms.double(0.0),
    coneRadius = cms.double(0.5),
    jetType = cms.untracked.string('PFJet'),
    inputEMin = cms.double(0.0),
    towerThreshold = cms.double(0.5)
)

Fastjet10PFJets = cms.EDProducer("FastJetProducer",
    FastjetParameters,
    src = cms.InputTag("particleFlowJetCandidates"),
    inputEtMin = cms.double(0.0),
    inputEMin = cms.double(0.0),
    FJ_ktRParam = cms.double(1.0),
    jetType = cms.untracked.string('PFJet'),
    towerThreshold = cms.double(0.5),
    PtMin = cms.double(1.0)
)


