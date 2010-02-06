import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("USER")
process.maxEvents = cms.untracked.PSet(
      #input = cms.untracked.int32(-1)
      input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
      debugVerbosity = cms.untracked.uint32(0),
      debugFlag = cms.untracked.bool(False),
      fileNames = cms.untracked.vstring("file:/data4/Wmunu-Summer09-MC_31X_V2_preproduction_311-v1/0011/F4C91F77-766D-DE11-981F-00163E1124E7.root")
)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('corMetWMuNus'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
            threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

# Producers --> Create one collection of WMuNus per met type
  
process.pfMetWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("pfMet"),
      METIncludesMuons = cms.untracked.bool(True),
      JetTag = cms.untracked.InputTag("sisCone5CaloJets"),
      
      # Preselection! 
      ApplyPreselection = cms.untracked.bool(True),
      MuonTrig = cms.untracked.string("HLT_Mu9"),
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),
)

process.tcMetWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("tcMet"),
      METIncludesMuons = cms.untracked.bool(True),
      JetTag = cms.untracked.InputTag("sisCone5CaloJets"),

      # Preselection!
      ApplyPreselection = cms.untracked.bool(True),
      MuonTrig = cms.untracked.string("HLT_Mu9"),
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),
)

process.corMetWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("corMetGlobalMuons"),
      METIncludesMuons = cms.untracked.bool(True),
      JetTag = cms.untracked.InputTag("sisCone5CaloJets"),

      # Preselection!
      ApplyPreselection = cms.untracked.bool(True),
      MuonTrig = cms.untracked.string("HLT_Mu9"),
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),
)

# Output
process.load("Configuration.EventContent.EventContent_cff")

process.myEventContent = cms.PSet(
	outputCommands = cms.untracked.vstring(
		'keep *'
	)
)

process.wmnOutput = cms.OutputModule("PoolOutputModule",
      #process.AODSIMEventContent,
      process.myEventContent,
      SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring('path1','path2','path3')
      ),
      fileName = cms.untracked.string('AOD_with_WCandidates.root')
)


# Steering the process
process.path1 = cms.Path(process.corMetWMuNus)
process.path2 = cms.Path(process.pfMetWMuNus)
process.path3 = cms.Path(process.tcMetWMuNus)
process.end = cms.EndPath(process.wmnOutput)



