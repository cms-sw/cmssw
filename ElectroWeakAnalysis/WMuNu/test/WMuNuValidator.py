import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("wmnhist")
process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
      #input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
      debugVerbosity = cms.untracked.uint32(0),
      debugFlag = cms.untracked.bool(False),
      fileNames = cms.untracked.vstring("file:/data4/RelValWM_CMSSW_3_1_0-STARTUP31X_V1-v1_GEN-SIM-RECO/40BFAA1A-5466-DE11-B792-001D09F29533.root")
)

# Printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('wmnHistBeforeCuts','wmnSelFilter','wmnHistAfterCuts'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
            threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

# Histograms before selection
process.wmnHistBeforeCuts = cms.EDAnalyzer("WMuNuValidator",
      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("met"),
      METIncludesMuons = cms.untracked.bool(False),
      JetTag = cms.untracked.InputTag("sisCone5CaloJets"),
      # Main cuts ->
      MuonTrig = cms.untracked.string("HLT_Mu9"),
      IsCombinedIso = cms.untracked.bool(False),
      NJetMax = cms.untracked.int32(999999)
)

# Selector and parameters
process.wmnSelFilter = cms.EDFilter("WMuNuSelector",
      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("met"),
      METIncludesMuons = cms.untracked.bool(False),
      JetTag = cms.untracked.InputTag("sisCone5CaloJets"),
      
      # Main cuts ->
      MuonTrig = cms.untracked.string("HLT_Mu9"),
      UseTrackerPt = cms.untracked.bool(True),
      PtCut = cms.untracked.double(25.0),
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(False),
      IsoCut03 = cms.untracked.double(0.1),
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(200.0),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(2.),

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2),
      NormalizedChi2Cut = cms.untracked.double(10.),
      TrackerHitsCut = cms.untracked.int32(11),
      IsAlsoTrackerMuon = cms.untracked.bool(True),
      
      # To suppress Zmm ->
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      
      # To further suppress ttbar ->
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999)
)

# Histograms after selection
process.wmnHistAfterCuts = cms.EDAnalyzer("WMuNuValidator",
      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("met"),
      METIncludesMuons = cms.untracked.bool(False),
      JetTag = cms.untracked.InputTag("sisCone5CaloJets"),
      # Main cuts ->
      MuonTrig = cms.untracked.string("HLT_Mu9"),
      IsCombinedIso = cms.untracked.bool(False),
      NJetMax = cms.untracked.int32(999999)
)

# Output events
process.load("Configuration.EventContent.EventContent_cff")
process.wmnOutput = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('wmnhist')
    ),
    fileName = cms.untracked.string('root_files/WMuNu_events.root')
)

# Output histograms
process.TFileService = cms.Service("TFileService", fileName = cms.string('WMuNu_histograms.root') )

# Steering the process
process.wmnhist = cms.Path(process.wmnHistBeforeCuts+process.wmnSelFilter+process.wmnHistAfterCuts)
process.end = cms.EndPath(process.wmnOutput)
