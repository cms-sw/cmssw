import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *


pfClusterRefsForJetsHCAL = cms.EDProducer("PFClusterRefCandidateProducer",
    src          = cms.InputTag('particleFlowClusterHCAL'),
    particleType = cms.string('pi+')
)

pfClusterRefsForJetsECAL = cms.EDProducer("PFClusterRefCandidateProducer",
    src          = cms.InputTag('particleFlowClusterECAL'),
   # src          = cms.InputTag('particleFlowCluster'),
    particleType = cms.string('pi+')
)

pfClusterRefsForJetsHF = cms.EDProducer("PFClusterRefCandidateProducer",
    src          = cms.InputTag('particleFlowClusterHF'),
    particleType = cms.string('pi+')
)

pfClusterRefsForJetsHO = cms.EDProducer("PFClusterRefCandidateProducer",
    src          = cms.InputTag('particleFlowClusterHO'),
    particleType = cms.string('pi+')
)


pfClusterRefsForJets = cms.EDProducer("PFClusterRefCandidateMerger",
    src = cms.VInputTag("pfClusterRefsForJetsHCAL", "pfClusterRefsForJetsECAL", "pfClusterRefsForJetsHF", "pfClusterRefsForJetsHO")
#    src = cms.VInputTag("pfClusterRefsForJetsHCAL", "pfClusterRefsForJetsECAL","pfClusterRefsForJetsHF")
)

pfClusterRefsForJets_step = cms.Sequence(
   particleFlowRecHitECAL*
   particleFlowRecHitHBHE*
   particleFlowRecHitHF*
   particleFlowRecHitHO*
   particleFlowClusterECALUncorrected*
   particleFlowClusterECAL*
   particleFlowClusterHBHE*
   particleFlowClusterHCAL*
   particleFlowClusterHF*
   particleFlowClusterHO*

   pfClusterRefsForJetsHCAL*
   pfClusterRefsForJetsECAL*
   pfClusterRefsForJetsHF*
   pfClusterRefsForJetsHO*
   pfClusterRefsForJets
)
