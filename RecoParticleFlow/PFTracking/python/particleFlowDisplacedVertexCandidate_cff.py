import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.particleFlowDisplacedVertexCandidate_cfi import *

from FWCore.MessageLogger.MessageLogger_cfi import *
MessageLogger.suppressWarning.extend(cms.untracked.vstring("particleFlowDisplacedVertexCandidate"));


