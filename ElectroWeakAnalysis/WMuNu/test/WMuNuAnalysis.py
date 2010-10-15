import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("USER")
process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
)

process.load("ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi")

process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring(
     ),
      inputCommands = cms.untracked.vstring(
      'keep *', 'drop *_lumiProducer_*_*', 'drop *_MEtoEDMConverter_*_*', 'drop *_l1GtTriggerMenuLite_*_*' , 'drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT*','drop edmTriggerResults_TriggerResults__makeSD','drop *_*__JuntandoSkims','drop *_*__SuperMuSkim' 
      )
)



process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('corMet','pfMet','tcMet'),
      cout = cms.untracked.PSet(
             default = cms.untracked.PSet( limit = cms.untracked.int32(100) ),
             threshold = cms.untracked.string('INFO')
#             threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

process.corMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("corMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(999.),
      PtCut = cms.untracked.double(20.),
      MuonTrig=cms.untracked.string("HLT_Mu9"),
      SelectByCharge=cms.untracked.int32(0),
#      IsoCut03 = cms.untracked.double(1)
      TrigTag = cms.untracked.InputTag("TriggerResults::REDIGI36X")

)


process.pfMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("pfMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(999.),
      PtCut = cms.untracked.double(20.),
      MuonTrig=cms.untracked.string("HLT_Mu9"),
      SelectByCharge=cms.untracked.int32(0),
#      IsoCut03 = cms.untracked.double(1)      
      TrigTag = cms.untracked.InputTag("TriggerResults::REDIGI36X")

)


process.tcMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("tcMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5CaloJets"),
      AcopCut = cms.untracked.double(999.),
      PtCut = cms.untracked.double(20.),
      MuonTrig=cms.untracked.string("HLT_Mu9"),
      SelectByCharge=cms.untracked.int32(0),
#      IsoCut03 = cms.untracked.double(1),
      TrigTag = cms.untracked.InputTag("TriggerResults::REDIGI36X")
)

process.load("ElectroWeakAnalysis.WMuNu.wmunusValidation_cfi") #load validation sequence (for WMunu & ZMuMu)
process.wmnVal_pfMet.JetTag = cms.untracked.InputTag("ak5CaloJets")
process.wmnVal_corMet.JetTag = cms.untracked.InputTag("ak5CaloJets") 
process.wmnVal_tcMet.JetTag = cms.untracked.InputTag("ak5CaloJets") 
process.wmnVal_tcMet.PtCut =  cms.untracked.double(20.)
process.wmnVal_pfMet.PtCut =  cms.untracked.double(20.)
process.wmnVal_corMet.PtCut =  cms.untracked.double(20.)
process.wmnVal_tcMet.MuonTrig =  cms.untracked.string("HLT_Mu9")
process.wmnVal_pfMet.MuonTrig =  cms.untracked.string("HLT_Mu9")
process.wmnVal_tcMet.MuonTrig =  cms.untracked.string("HLT_Mu9")
process.wmnVal_pfMet.MuonTrig =  cms.untracked.string("HLT_Mu9")
process.wmnVal_corMet.MuonTrig =  cms.untracked.string("HLT_Mu9")
process.wmnVal_tcMet.AcopCut =  cms.untracked.double(999.)
process.wmnVal_pfMet.AcopCut =  cms.untracked.double(999.)
process.wmnVal_corMet.AcopCut =  cms.untracked.double(999.)
process.wmnVal_corMet.TrigTag = cms.untracked.InputTag("TriggerResults::REDIGI36X")
process.wmnVal_pfMet.TrigTag = cms.untracked.InputTag("TriggerResults::REDIGI36X")
process.wmnVal_tcMet.TrigTag = cms.untracked.InputTag("TriggerResults::REDIGI36X")


process.TFileService = cms.Service("TFileService", fileName = cms.string('Wmunu.root') )

#process.eventDump = cms.EDAnalyzer(
#    "EventDumper",
 #    srcMuons = cms.InputTag("goodMuonsPt15")
#    )

# Steering the process
process.path1 = cms.Path(process.wmnVal_corMet)
process.path2 = cms.Path(process.wmnVal_pfMet)
process.path3 = cms.Path(process.wmnVal_tcMet)

process.path5 = cms.Path(process.corMetWMuNus+process.corMet)
process.path6 = cms.Path(process.pfMetWMuNus+process.pfMet)
process.path7 = cms.Path(process.tcMetWMuNus+process.tcMet)






