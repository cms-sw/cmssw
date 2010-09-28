import FWCore.ParameterSet.Config as cms

zjetsRECO = cms.untracked.vstring(
     '/store/mc/Spring10/ZJets-madgraph/AODSIM/START3X_V26_S09-v1/0013/00EFC4EA-3847-DF11-A194-003048D4DF80.root'
    ,'/store/mc/Spring10/ZJets-madgraph/AODSIM/START3X_V26_S09-v1/0013/0C096217-3A47-DF11-9E65-003048C692A4.root'
)

zjetsTracks  = cms.untracked.vstring(
    '/store/user/rwolf/school/patTuple_zjets_tracks.root'
)

zjetsTrigger  = cms.untracked.vstring(
    '/store/user/rwolf/school/patTuple_zjets_trigger.root'
)
