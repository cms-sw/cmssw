import FWCore.ParameterSet.Config as cms


ttbarRECO = cms.untracked.vstring(    
     '/store/relval/CMSSW_3_8_4/RelValZTT/GEN-SIM-RECO/START38_V12-v1/0024/46893A0C-82C2-DF11-B8A3-003048678BAE.root'
    ,'/store/relval/CMSSW_3_8_4/RelValZTT/GEN-SIM-RECO/START38_V12-v1/0024/8E51890C-82C2-DF11-B7ED-002618943949.root'
)

ttbarJets  = cms.untracked.vstring(
    'rfio:///castor/cern.ch/user/c/cmssup/patTuple_ttbar_jets.root'
)

zjetsRECO = cms.untracked.vstring(
     '/store/relval/CMSSW_3_8_4/RelValZMM/GEN-SIM-RECO/START38_V12-v1/0024/00ABC27B-86C2-DF11-A82D-003048678B16.root'
    ,'/store/relval/CMSSW_3_8_4/RelValZMM/GEN-SIM-RECO/START38_V12-v1/0024/42595C94-7FC2-DF11-A952-003048678FE4.root'
)

zjetsTracks  = cms.untracked.vstring(
    'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_tracks.root'
)

zjetsTrigger  = cms.untracked.vstring(
    'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_trigger.root'
)


