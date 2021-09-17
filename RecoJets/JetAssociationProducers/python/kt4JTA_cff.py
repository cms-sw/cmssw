import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
kt4JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("kt4CaloJets")
)

kt4JetTracksAssociatorAtCaloFace = cms.EDProducer("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("kt4CaloJets")
)

kt4JetExtender = cms.EDProducer("JetExtender",
    jets = cms.InputTag("kt4CaloJets"),
    jet2TracksAtCALO = cms.InputTag("kt4JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("kt4JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

kt4JTATask = cms.Task(kt4JetTracksAssociatorAtVertex,
                      kt4JetTracksAssociatorAtCaloFace,
                      kt4JetExtender)
kt4JTA = cms.Sequence(kt4JTATask)
