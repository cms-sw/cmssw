import FWCore.ParameterSet.Config as cms

# $Id: midPointCone5JTA_cff.py,v 1.2 2008/04/21 03:27:51 rpw Exp $
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
midPointCone5JetTracksAssociatorAtVertex = cms.EDFilter("JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("midPointCone5CaloJets")
)

midPointCone5JetTracksAssociatorAtCaloFace = cms.EDFilter("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO,
    jets = cms.InputTag("midPointCone5CaloJets")
)

midPointCone5JetExtender = cms.EDFilter("JetExtender",
    jets = cms.InputTag("midPointCone5CaloJets"),
    jet2TracksAtCALO = cms.InputTag("midPointCone5JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("midPointCone5JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

midPointCone5JTA = cms.Sequence(midPointCone5JetTracksAssociatorAtVertex*midPointCone5JetTracksAssociatorAtCaloFace*midPointCone5JetExtender)

