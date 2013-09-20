import FWCore.ParameterSet.Config as cms

# $Id: ak5JTA_cff.py,v 1.4 2012/07/11 14:12:14 srappocc Exp $
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ak5JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak5CaloJets")
)

ak5JetTracksAssociatorAtVertexPF = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak5PFJetsCHS")
)


ak5JetTracksAssociatorExplicit = cms.EDProducer("JetTracksAssociatorExplicit",
    j2tParametersVX,
    jets = cms.InputTag("ak5PFJetsCHS")
)

ak5JetTracksAssociatorAtCaloFace = cms.EDProducer("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("ak5CaloJets")
)

ak5JetExtender = cms.EDProducer("JetExtender",
    jets = cms.InputTag("ak5CaloJets"),
    jet2TracksAtCALO = cms.InputTag("ak5JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("ak5JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

ak5JTA = cms.Sequence(ak5JetTracksAssociatorAtVertexPF*
                      ak5JetTracksAssociatorAtVertex*
                      ak5JetTracksAssociatorAtCaloFace*ak5JetExtender)

ak5JTAExplicit = cms.Sequence(ak5JetTracksAssociatorExplicit)
