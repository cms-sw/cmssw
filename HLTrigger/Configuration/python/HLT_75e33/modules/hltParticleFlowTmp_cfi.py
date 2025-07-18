import FWCore.ParameterSet.Config as cms

hltParticleFlowTmp = cms.EDProducer("PFCandidateListMerger",
    src = cms.VInputTag("hltParticleFlowTmpBarrel", "hltPfTICL")
)
