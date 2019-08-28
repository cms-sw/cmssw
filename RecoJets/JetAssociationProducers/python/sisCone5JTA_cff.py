import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
sisCone5JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("sisCone5CaloJets")
)

sisCone5JetTracksAssociatorAtCaloFace = cms.EDProducer("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("sisCone5CaloJets")
)

sisCone5JetExtender = cms.EDProducer("JetExtender",
    jets = cms.InputTag("sisCone5CaloJets"),
    jet2TracksAtCALO = cms.InputTag("sisCone5JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("sisCone5JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

sisCone5JTATask = cms.Task(sisCone5JetTracksAssociatorAtVertex,
                           sisCone5JetTracksAssociatorAtCaloFace,
                           sisCone5JetExtender)
sisCone5JTA = cms.Sequence(sisCone5JTATask)
