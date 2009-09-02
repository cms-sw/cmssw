import FWCore.ParameterSet.Config as cms

# $Id: ak7JTA_cff.py,v 1.1 2009/07/31 04:01:53 srappocc Exp $
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ak7JetTracksAssociatorAtVertex = cms.EDFilter("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("ak7CaloJets")
)

ak7JetTracksAssociatorAtCaloFace = cms.EDFilter("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("ak7CaloJets")
)

ak7JetExtender = cms.EDFilter("JetExtender",
    jets = cms.InputTag("ak7CaloJets"),
    jet2TracksAtCALO = cms.InputTag("ak7JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("ak7JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.7)
)

ak7JTA = cms.Sequence(ak7JetTracksAssociatorAtVertex*ak7JetTracksAssociatorAtCaloFace*ak7JetExtender)

