import FWCore.ParameterSet.Config as cms

# Standard Iterative Cone Jets parameters
# $Id
InsideOutConeJetParameters = cms.PSet(
    seedObjectPt     = cms.double(5.0),
    growthParameter  = cms.double(5.0),
    maxSize          = cms.double(0.4),
    minSize          = cms.double(0.01),
    debugLevel  = cms.untracked.int32(0)
)

from RecoJets.JetProducers.PFJetParameters_cfi import *

insideOutCone5PFJets = cms.EDProducer("CMSInsideOutProducer",
    InsideOutConeJetParameters,
    PFJetParameters,
    alias = cms.untracked.string('InsideOut4PFJet'),
)

# the jet tracks association producer, needed for RecoTau algorithm
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *

insideOutJetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
      j2tParametersVX,
      jets = cms.InputTag("insideOutCone5PFJets")
)

insideOutJets = cms.Sequence(
      insideOutCone5PFJets*
      insideOutJetTracksAssociatorAtVertex
)

