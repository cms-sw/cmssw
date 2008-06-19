import FWCore.ParameterSet.Config as cms

# $Id: kt4JTA_cff.py,v 1.2 2008/04/21 03:27:49 rpw Exp $
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
kt4JetTracksAssociatorAtVertex = cms.EDFilter("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("kt4CaloJets")
)

kt4JetTracksAssociatorAtCaloFace = cms.EDFilter("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("kt4CaloJets")
)

kt4JetExtender = cms.EDFilter("JetExtender",
    jets = cms.InputTag("kt4CaloJets"),
    jet2TracksAtCALO = cms.InputTag("kt4JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("kt4JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

kt4JTA = cms.Sequence(kt4JetTracksAssociatorAtVertex*kt4JetTracksAssociatorAtCaloFace*kt4JetExtender)

