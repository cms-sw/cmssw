import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

pfClusterRefsForJetsHCAL = cms.EDProducer("PFClusterRefCandidateProducer",
    src          = cms.InputTag('particleFlowClusterHCAL'),
    particleType = cms.string('pi+')
)

pfClusterRefsForJetsECAL = cms.EDProducer("PFClusterRefCandidateProducer",
    src          = cms.InputTag('particleFlowClusterECAL'),
    particleType = cms.string('pi+')
)

pfClusterRefsForJetsHFEM = cms.EDProducer("PFClusterRefCandidateProducer",
    src          = cms.InputTag('particleFlowClusterHFEM'),
    particleType = cms.string('pi+')
)

pfClusterRefsForJetsHFHAD = cms.EDProducer("PFClusterRefCandidateProducer",
    src          = cms.InputTag('particleFlowClusterHFHAD'),
    particleType = cms.string('pi+')
)

pfClusterRefsForJets = cms.EDProducer("PFClusterRefCandidateMerger",
    src = cms.VInputTag("pfClusterRefsForJetsHCAL", "pfClusterRefsForJetsECAL")
#    src = cms.VInputTag("pfClusterRefsForJetsHCAL", "pfClusterRefsForJetsECAL","pfClusterRefsForJetsHFEM","pfClusterRefsForJetsHFHAD")
)


