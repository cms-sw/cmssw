import FWCore.ParameterSet.Config as cms

# $Id: ak4JTA_cff.py,v 1.3 2012/01/13 21:11:04 srappocc Exp $
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ak4JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak4CaloJets")
)

ak4JetTracksAssociatorExplicit = cms.EDProducer("JetTracksAssociatorExplicit",
    j2tParametersVX,
    jets = cms.InputTag("ak4PFJets")
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

ak4JTA = cms.Sequence(ak4JetTracksAssociatorAtVertex*ak4JetTracksAssociatorAtCaloFace*ak4JetExtender)

ak4JTAExplicit = cms.Sequence(ak4JetTracksAssociatorExplicit)
