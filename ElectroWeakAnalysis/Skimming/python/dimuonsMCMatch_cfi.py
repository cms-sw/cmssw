import FWCore.ParameterSet.Config as cms

dimuonsMCMatch = cms.EDProducer("MCTruthCompositeMatcherNew",
    src = cms.InputTag("dimuons"),
    #
    # comment PAT match because works only for layer-0 muons  
    #
    #  VInputTag matchMaps = { muonMatch }
    matchPDGId = cms.vint32(23),
    matchMaps = cms.VInputTag(cms.InputTag("goodMuonMCMatch"))
)


