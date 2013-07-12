import FWCore.ParameterSet.Config as cms

# $Id: ak7JTA_cff.py,v 1.1 2009/09/02 03:00:44 srappocc Exp $
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ak7JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak7CaloJets")
)

ak7JetTracksAssociatorAtCaloFace = cms.EDProducer("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("ak7CaloJets")
)

ak7JetExtender = cms.EDProducer("JetExtender",
    jets = cms.InputTag("ak7CaloJets"),
    jet2TracksAtCALO = cms.InputTag("ak7JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("ak7JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.7)
)

ak7JTA = cms.Sequence(ak7JetTracksAssociatorAtVertex*ak7JetTracksAssociatorAtCaloFace*ak7JetExtender)

