import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ak8JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak8CaloJets")
)

ak8JetTracksAssociatorAtVertexPF = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak8PFJetsCHS")
)


ak8JetTracksAssociatorExplicit = cms.EDProducer("JetTracksAssociatorExplicit",
    j2tParametersVX,
    jets = cms.InputTag("ak8PFJetsCHS")
)

ak8JetTracksAssociatorAtCaloFace = cms.EDProducer("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("ak8CaloJets")
)

ak8JetExtender = cms.EDProducer("JetExtender",
    jets = cms.InputTag("ak8CaloJets"),
    jet2TracksAtCALO = cms.InputTag("ak8JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("ak8JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.8)
)

ak8JTA = cms.Sequence(ak8JetTracksAssociatorAtVertexPF*
                      ak8JetTracksAssociatorAtVertex*
                      ak8JetTracksAssociatorAtCaloFace*ak8JetExtender)

ak8JTAExplicit = cms.Sequence(ak8JetTracksAssociatorExplicit)
