import FWCore.ParameterSet.Config as cms

# $Id: iterativeCone5JTA_cff.py,v 1.4 2010/02/17 17:47:53 wmtan Exp $
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
iterativeCone5JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("iterativeCone5CaloJets")
)

iterativeCone5JetTracksAssociatorAtCaloFace = cms.EDProducer("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("iterativeCone5CaloJets")
)

iterativeCone5JetExtender = cms.EDProducer("JetExtender",
    jets = cms.InputTag("iterativeCone5CaloJets"),
    jet2TracksAtCALO = cms.InputTag("iterativeCone5JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("iterativeCone5JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

iterativeCone5JTA = cms.Sequence(iterativeCone5JetTracksAssociatorAtVertex*iterativeCone5JetTracksAssociatorAtCaloFace*iterativeCone5JetExtender)

