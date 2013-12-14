import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoMET.METProducers.METSigParams_cfi import *

##____________________________________________________________________________||
pfMet = cms.EDProducer(
    "PFMETProducer",
    METSignificance_params,
    src = cms.InputTag("particleFlow"),
    alias = cms.string('PFMET'),
    globalThreshold = cms.double(0.0),
    calculateSignificance = cms.bool(True),
    jets = cms.InputTag("ak4PFJets") # for significance calculation
    )
##____________________________________________________________________________||

