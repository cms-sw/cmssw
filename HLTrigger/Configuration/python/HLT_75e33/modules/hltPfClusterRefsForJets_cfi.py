import FWCore.ParameterSet.Config as cms

hltPfClusterRefsForJets = cms.EDProducer("PFClusterRefCandidateMerger",
    src = cms.VInputTag("hltPfClusterRefsForJetsHCAL", "hltPfClusterRefsForJetsECAL", "hltPfClusterRefsForJetsHF", "hltPfClusterRefsForJetsHO", "hltPfClusterRefsForJetsHGCAL")
)
