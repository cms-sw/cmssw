import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoMET.METProducers.METSigParams_cfi import *

##____________________________________________________________________________||
pfClusterMet = cms.EDProducer(
    "PFClusterMETProducer",
    src = cms.InputTag("pfClusterRefsForJets"),
    alias = cms.string('pfClusterMet'),
    globalThreshold = cms.double(0.0),
    )

##____________________________________________________________________________||
