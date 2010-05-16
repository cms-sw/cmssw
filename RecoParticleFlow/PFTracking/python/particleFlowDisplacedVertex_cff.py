import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.particleFlowDisplacedVertex_cfi import *

from FWCore.MessageLogger.MessageLogger_cfi import *
#MessageLogger.suppressWarning = cms.untracked.vstring("particleFlowDisplacedVertexCandidate", "particleFlowDisplacedVertex");
MessageLogger.suppressWarning.extend(cms.untracked.vstring("particleFlowDisplacedVertex"));
