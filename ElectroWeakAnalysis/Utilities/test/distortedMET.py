import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("distortMET")
process.maxEvents = cms.untracked.PSet(
      #input = cms.untracked.int32(-1)
      input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
      debugVerbosity = cms.untracked.uint32(0),
      debugFlag = cms.untracked.bool(False),
      fileNames = cms.untracked.vstring("file:/data4/Wmunu-Summer09-MC_31X_V2_preproduction_311-v1/0011/F4C91F77-766D-DE11-981F-00163E1124E7.root")
)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('distortedMET','wmnSelFilter'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
            threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

# Selector and parameters
process.distortedMET = cms.EDFilter("DistortedMETProducer",
      MetTag = cms.untracked.InputTag("met"),
      MetScaleShift = cms.untracked.double(0.1)
)

process.wmnSelFilter = cms.EDFilter("WMuNuValidator",
      # Fast selection flag (no histograms or book-keeping) ->
      FastOption = cms.untracked.bool(True),

      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("distortedMET"),
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

# Output
#process.load("Configuration.EventContent.EventContent_cff")
#process.AODSIMEventContent.outputCommands.append('keep *_distortedMET_*_*')
#process.myEventContent = process.AODSIMEventContent
process.myEventContent = cms.PSet(
  outputCommands = cms.untracked.vstring(
      #'keep *_distortedMET_*_*',
      'keep *'
  )
)

process.wmnOutput = cms.OutputModule("PoolOutputModule",
      process.myEventContent,
      SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring('distortMET')
      ),
      fileName = cms.untracked.string('root_files/distortedMuons.root')
)

# Output histograms
#process.TFileService = cms.Service("TFileService", fileName = cms.string('WMuNu_histograms.root') )

# Steering the process
process.distortMET = cms.Path(process.distortedMET*process.wmnSelFilter)
process.end = cms.EndPath(process.wmnOutput)
