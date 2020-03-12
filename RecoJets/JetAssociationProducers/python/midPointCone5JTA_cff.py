import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
midPointCone5JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("midPointCone5CaloJets")
)

midPointCone5JetTracksAssociatorAtCaloFace = cms.EDProducer("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("midPointCone5CaloJets")
)

midPointCone5JetExtender = cms.EDProducer("JetExtender",
    jets = cms.InputTag("midPointCone5CaloJets"),
    jet2TracksAtCALO = cms.InputTag("midPointCone5JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("midPointCone5JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

midPointCone5JTATask = cms.Task(midPointCone5JetTracksAssociatorAtVertex,
                                midPointCone5JetTracksAssociatorAtCaloFace,
                                midPointCone5JetExtender)
midPointCone5JTA = cms.Sequence(midPointCone5JTATask)
