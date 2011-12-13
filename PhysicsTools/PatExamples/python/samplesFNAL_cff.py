import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

## 299,991 QCD events as defined on WorkBookPATExampleTopQuarks
simulationQCD = cms.untracked.vstring(
     'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_0.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_1.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_2.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_3.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_4.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_5.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_6.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_7.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_8.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_9.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_10.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_11.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_12.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_13.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_14.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_15.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_16.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_17.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_18.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_19.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_2.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_21.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_22.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_23.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_24.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_25.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_26.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_27.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_28.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_qcd_29.root'
)

##  99,991 W+Jets events as defined on WorkBookPATExampleTopQuarks
simulationWjets = cms.untracked.vstring(
     'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_wjets_0.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_wjets_1.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_wjets_2.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_wjets_3.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_wjets_4.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_wjets_5.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_wjets_6.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_wjets_7.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_wjets_8.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_wjets_9.root'
    )

##   9,991 Z+Jets events as defined on WorkBookPATExampleTopQuarks
simulationZjets = cms.untracked.vstring(
     'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_zjets_0.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_zjets_1.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_zjets_2.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_zjets_3.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_zjets_4.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_zjets_5.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_zjets_6.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_zjets_7.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_zjets_8.root'
    ,'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_zjets_9.root'
)

##   1000 Ttbar events as defined on WorkBookPATExampleTopQuarks
simulationTtbar = cms.untracked.vstring(
    'file:/uscms_data/d3/rwolf/PATTutorial/Nov10/patTuple_ttbar.root'
)

zjetsTrigger  = cms.untracked.vstring(
    'file:/uscms_data/d3/vadler/PATTutorial/Nov10/patTuple_zjets_trigger.root'
)

#zjetsRECO = cms.untracked.vstring(
#    pickRelValInputFiles( relVal = 'RelValZMM' )
#)

#ttbarRECO = cms.untracked.vstring(
#    pickRelValInputFiles( relVal = 'RelValZTT' )
#)

# CMSSW_3_8_6 re-reconstruction of muon PD, run 144112, 17717 events AOD
dataMu = cms.untracked.vstring(
    '/store/data/Run2010A/Mu/AOD/Nov4ReReco_v1/0011/D2E5D86F-AEEC-DF11-B261-0017A4771028.root'
)
