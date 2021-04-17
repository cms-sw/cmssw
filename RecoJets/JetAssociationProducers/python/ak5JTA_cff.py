import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * ##propagator

from RecoJets.JetAssociationProducers.j2tParametersCALO_cfi import *
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
ak5JetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX.clone(coneSize = cms.double(0.5)),
    jets = cms.InputTag("ak5CaloJets")
)

ak5JetTracksAssociatorAtVertexPF = cms.EDProducer("JetTracksAssociatorAtVertex",
    j2tParametersVX.clone(coneSize = cms.double(0.5)),
    jets = cms.InputTag("ak5PFJetsCHS")
)


ak5JetTracksAssociatorExplicit = cms.EDProducer("JetTracksAssociatorExplicit",
    j2tParametersVX.clone(coneSize = cms.double(0.5)),
    jets = cms.InputTag("ak5PFJetsCHS")
)

ak5JetTracksAssociatorAtCaloFace = cms.EDProducer("JetTracksAssociatorAtCaloFace",
    j2tParametersCALO.clone(coneSize = cms.double(0.5)),
    jets = cms.InputTag("ak5CaloJets")
)

ak5JetExtender = cms.EDProducer("JetExtender",
    jets = cms.InputTag("ak5CaloJets"),
    jet2TracksAtCALO = cms.InputTag("ak5JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("ak5JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

ak5JTATask = cms.Task(ak5JetTracksAssociatorAtVertexPF,
                      ak5JetTracksAssociatorAtVertex,
                      ak5JetTracksAssociatorAtCaloFace,
                      ak5JetExtender)
ak5JTA = cms.Sequence(ak5JTATask)

ak5JTAExplicitTask = cms.Task(ak5JetTracksAssociatorExplicit)
ak5JTAExplicit = cms.Sequence(ak5JTAExplicitTask)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(ak5JetTracksAssociatorAtVertex,
                  jets = "akCs4PFJets",
                  tracks = "highPurityGeneralTracks"
)
