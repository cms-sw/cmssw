import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ak7JetTracksAssociatorAtVertexPF = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX.clone(coneSize = cms.double(0.5)),
    jets = cms.InputTag("ak7PFJetsCHS")
)

ak7JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX.clone(coneSize = cms.double(0.5)),
    jets = cms.InputTag("ak7CaloJets")
)

ak7JetTracksAssociatorAtCaloFace = cms.EDProducer("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO.clone(coneSize = cms.double(0.5)),
    jets = cms.InputTag("ak7CaloJets")
)

ak7JetExtender = cms.EDProducer("JetExtender",
    jets = cms.InputTag("ak7CaloJets"),
    jet2TracksAtCALO = cms.InputTag("ak7JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("ak7JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.7)
)

ak7JTATask = cms.Task(ak7JetTracksAssociatorAtVertexPF,
                      ak7JetTracksAssociatorAtVertex,
                      ak7JetTracksAssociatorAtCaloFace,
                      ak7JetExtender)
ak7JTA = cms.Sequence(ak7JTATask)
