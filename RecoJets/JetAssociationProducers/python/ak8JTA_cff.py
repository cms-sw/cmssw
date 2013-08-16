import FWCore.ParameterSet.Config as cms

# $Id: ak8JTA_cff.py,v 1.1 2009/09/02 03:00:44 srappocc Exp $
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ak8JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak8CaloJets")
)

ak8JetTracksAssociatorAtCaloFace = cms.EDProducer("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("ak8CaloJets")
)

ak8JetExtender = cms.EDProducer("JetExtender",
    jets = cms.InputTag("ak8CaloJets"),
    jet2TracksAtCALO = cms.InputTag("ak8JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("ak8JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.7)
)

ak8JTA = cms.Sequence(ak8JetTracksAssociatorAtVertex*ak8JetTracksAssociatorAtCaloFace*ak8JetExtender)

