import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoMET.METProducers.METSigParams_cfi import *

##____________________________________________________________________________||
from RecoMET.METProducers.pfClusterMETProducer_cfi import pfClusterMETProducer as _pfClusterMETProducer
pfClusterMet = _pfClusterMETProducer.clone(
    src = "pfClusterRefsForJets",
    alias = 'pfClusterMet',
    globalThreshold = 0.0,
)

##____________________________________________________________________________||
