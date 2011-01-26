import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("USER")
process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
)

process.load("ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi")
process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring(
            "file:input.root"
     ),
)


process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('pfMet'),
      cout = cms.untracked.PSet(
             default = cms.untracked.PSet( limit = cms.untracked.int32(100) ),
             threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

process.pfMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      saveNTuple = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("pfMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5PFJets"),
      AcopCut = cms.untracked.double(999.),
      EtaCut = cms.untracked.double(2.1),
      PtCut = cms.untracked.double(25.),
      IsoCut03=cms.untracked.double(0.10),
      MuonTrig=cms.untracked.vstring("HLT_Mu9","HLT_Mu11","HLT_Mu15_v1"),
      SelectByCharge=cms.untracked.int32(0),
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),

)

process.tcMet = cms.EDFilter("WMuNuSelector",
      plotHistograms = cms.untracked.bool(True),
      saveNTuple = cms.untracked.bool(True),
      WMuNuCollectionTag = cms.untracked.InputTag("tcMetWMuNus"),
      JetTag = cms.untracked.InputTag("ak5PFJets"),
      AcopCut = cms.untracked.double(999.),
      EtaCut = cms.untracked.double(2.1),
      PtCut = cms.untracked.double(25.),
      IsoCut03=cms.untracked.double(0.10),
      MuonTrig=cms.untracked.vstring("HLT_Mu9","HLT_Mu11","HLT_Mu15_v1"),
      SelectByCharge=cms.untracked.int32(0),
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),

)

process.TFileService = cms.Service("TFileService", fileName = cms.string('WNTuple.root') )


process.path5 = cms.Path(process.pfMetWMuNus+process.pfMet)
process.path6 = cms.Path(process.tcMetWMuNus+process.tcMet)






