import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ak4JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak4CaloJets")
)

ak4JetTracksAssociatorAtVertexPF = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak4PFJetsCHS")
)


ak4JetTracksAssociatorExplicit = cms.EDProducer("JetTracksAssociatorExplicit",
    j2tParametersVX,
    jets = cms.InputTag("ak4PFJetsCHS")
)

ak4JetTracksAssociatorAtCaloFace = cms.EDProducer("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("ak4CaloJets")
)

ak4JetExtender = cms.EDProducer("JetExtender",
    jets = cms.InputTag("ak4CaloJets"),
    jet2TracksAtCALO = cms.InputTag("ak4JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("ak4JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.4)
)

ak4JTATask = cms.Task(ak4JetTracksAssociatorAtVertexPF,
                      ak4JetTracksAssociatorAtVertex,
                      ak4JetTracksAssociatorAtCaloFace,ak4JetExtender)
ak4JTA = cms.Sequence(ak4JTATask)

ak4JTAExplicitTask = cms.Task(ak4JetTracksAssociatorExplicit)
ak4JTAExplicit = cms.Sequence(ak4JTAExplicitTask)
