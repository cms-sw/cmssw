import FWCore.ParameterSet.Config as cms

# $Id: ak5JTA_cff.py,v 1.3 2008/06/19 14:55:14 rahatlou Exp $
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ak5JetTracksAssociatorAtVertex = cms.EDFilter("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak5CaloJets")
)

ak5JetTracksAssociatorAtCaloFace = cms.EDFilter("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("ak5CaloJets")
)

ak5JetExtender = cms.EDFilter("JetExtender",
    jets = cms.InputTag("ak5CaloJets"),
    jet2TracksAtCALO = cms.InputTag("ak5JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("ak5JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

ak5JTA = cms.Sequence(ak5JetTracksAssociatorAtVertex*ak5JetTracksAssociatorAtCaloFace*ak5JetExtender)

