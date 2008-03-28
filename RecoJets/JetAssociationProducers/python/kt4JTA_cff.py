import FWCore.ParameterSet.Config as cms

# $Id: kt4JTA.cff,v 1.1 2007/10/26 22:26:03 fedor Exp $
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
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

