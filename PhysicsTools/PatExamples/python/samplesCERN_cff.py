import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

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
   pickRelValInputFiles( cmsswVersion  = 'CMSSW_4_1_3'
                       , relVal        = 'RelValZTT'
                       , globalTag     = 'START311_V2'
                       , numberOfFiles = 0 )
)

ttbarJets  = cms.untracked.vstring(
    'rfio:///castor/cern.ch/user/c/cmssup/patTuple_ttbar_jets.root'
)

zjetsRECO = cms.untracked.vstring(
   pickRelValInputFiles( cmsswVersion  = 'CMSSW_4_1_3'
                       , relVal        = 'RelValZMM'
                       , globalTag     = 'START311_V2'
                       , numberOfFiles = 0 )
)

zjetsTracks  = cms.untracked.vstring(
    'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_tracks.root'
)

zjetsTrigger  = cms.untracked.vstring(
    'rfio:///castor/cern.ch/user/c/cmssup/patTuple_zjets_trigger.root'
)

# CMSSW_3_8_5_patch3 prompt reconstruction of muon PD, run 149291, 22073 events AOD
dataMu = cms.untracked.vstring(
    '/store/data/Run2010B/Mu/AOD/PromptReco-v2/000/149/291/FE4109CA-D0E4-DF11-96F6-001D09F2AD7F.root'
)
