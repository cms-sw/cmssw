import FWCore.ParameterSet.Config as cms

particleFlowTmp = cms.EDProducer("PFCandidateListMerger",
    src = cms.VInputTag("particleFlowTmpBarrel", "pfTICL")
)
