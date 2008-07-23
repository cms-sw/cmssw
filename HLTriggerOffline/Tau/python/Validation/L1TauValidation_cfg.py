import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Tau.Validation.HLTTauReferences_cfi import *
from HLTriggerOffline.Tau.Validation.L1TauValidation_cfi import *


process = cms.Process("VALIDATEL1")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)


process.source = cms.Source("PoolSource",
                           fileNames = cms.untracked.vstring(
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/40FA3C45-E533-DD11-9B17-000423D98C20.root',
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/82A59E41-E533-DD11-BD1F-000423D9890C.root',
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/92ED629F-E833-DD11-AF71-001617DBD556.root',
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/9430E4AB-E833-DD11-A517-001617E30D12.root',
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/984CC643-E533-DD11-BA41-001617E30E2C.root',
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/9A551841-E533-DD11-82B0-000423D174FE.root',
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/B233AB12-EA33-DD11-82D5-000423D6CA02.root',
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/B8CACA72-E633-DD11-A861-001617C3B5F4.root',
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/BA2E0646-E533-DD11-9C7F-000423D98AF0.root',
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/CEEFCB50-E533-DD11-B759-000423D98DD4.root',
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/DAEAC6B8-E933-DD11-9807-001617E30CE8.root',
'/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/EAAE6B3F-E533-DD11-9B9C-000423D94908.root'
           )
)

process.load("FWCore.MessageService.MessageLogger_cfi")

process.DQMStore = cms.Service("DQMStore")

#Load 
process.load("HLTriggerOffline.Tau.Validation.HLTTauReferences_cfi")
process.load("HLTriggerOffline.Tau.Validation.L1TauValidation_cfi")

#Pick your favourite Boson (23 is Z , 25 is H0 , 35 is H+)
process.TauMCProducer.BosonID = 23
process.TauMCProducer.ptMinTau = 8.0
process.TauMCProducer.ptMinElectron = 4.0
process.TauMCProducer.ptMinMuon = 2.0

process.p1    = cms.Path(HLTTauRef+L1TauVal)







