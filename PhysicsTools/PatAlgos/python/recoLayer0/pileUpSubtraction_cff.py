import FWCore.ParameterSet.Config as cms

# PU SUB AND PARTICLES FOR ISO ---------------

from CommonTools.ParticleFlow.pfNoPileUp_cff import *
from CommonTools.ParticleFlow.pfParticleSelection_cff import *

# note pfPileUp modified according to JetMET's recommendations
pfPileUp.checkClosestZVertex = False
pfPileUp.Vertices = 'goodOfflinePrimaryVertices'
pfPileUp.PFCandidates = 'particleFlow'
pfNoPileUp.bottomCollection = 'particleFlow'

from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices
pfNoPileUpSequence.insert(0, goodOfflinePrimaryVertices)

PATPileUpSubtractionSequence = cms.Sequence(
    pfNoPileUpSequence +
    pfParticleSelectionSequence
    )

