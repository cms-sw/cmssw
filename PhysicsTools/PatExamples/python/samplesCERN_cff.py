import FWCore.ParameterSet.Config as cms

## 299,991 QCD events as defined on WorkBookPATExampleTopQuarks
simulationQCD = cms.untracked.vstring(
     'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_0.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_1.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_2.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_3.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_4.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_5.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_6.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_7.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_8.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_9.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_10.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_11.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_12.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_13.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_14.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_15.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_16.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_17.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_18.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_19.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_20.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_21.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_22.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_23.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_24.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_25.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_26.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_27.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_28.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_qcd_29.root'
)

##  99,991 W+Jets events as defined on WorkBookPATExampleTopQuarks
simulationWjets = cms.untracked.vstring(
     'rfio:///castor/cern.ch/user/c/cmssup/patTuple_wjets_0.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_wjets_1.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_wjets_2.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_wjets_3.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_wjets_4.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_wjets_5.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_wjets_6.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_wjets_7.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_wjets_8.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_wjets_9.root'
)

##   9,991 Z+Jets events as defined on WorkBookPATExampleTopQuarks
simulationZjets = cms.untracked.vstring(
     'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_0.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_1.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_2.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_3.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_4.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_5.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_6.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_7.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_8.root'
    ,'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_9.root'
)

##   1000 Ttbar events as defined on WorkBookPATExampleTopQuarks
simulationTtbar = cms.untracked.vstring(
     'rfio:///castor/cern.ch/user/c/cmssup/patTuple_ttbar.root'
)


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


