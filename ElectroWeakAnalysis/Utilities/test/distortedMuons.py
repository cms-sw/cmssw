import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("distortMuons")
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
      debugModules = cms.untracked.vstring('distortedMuons','wmnSelFilter','genMatchMap'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
            threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

process.genMatchMap = cms.EDFilter("MCTruthDeltaRMatcherNew",
    src = cms.InputTag("muons"),
    matched = cms.InputTag("genParticles"),
    distMin = cms.double(0.15),
    matchPDGId = cms.vint32(13)
)

# Selector and parameters
process.distortedMuons = cms.EDFilter("DistortedMuonProducer",
      MuonTag = cms.untracked.InputTag("muons"),
      GenMatchTag = cms.untracked.InputTag("genMatchMap"),
      EtaBinEdges = cms.untracked.vdouble(-2.1,2.1), # one more entry than next vectors
      MomentumScaleShift = cms.untracked.vdouble(1.e-3),
      UncertaintyOnOneOverPt = cms.untracked.vdouble(2.e-4), #in [1/GeV]
      RelativeUncertaintyOnPt = cms.untracked.vdouble(1.e-3),
      EfficiencyRatioOverMC = cms.untracked.vdouble(0.99)
)

process.wmnSelFilter = cms.EDFilter("WMuNuValidator",
      # Fast selection flag (no histograms or book-keeping) ->
      FastOption = cms.untracked.bool(True),

      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      MuonTag = cms.untracked.InputTag("distortedMuons"),
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

# Output
#process.load("Configuration.EventContent.EventContent_cff")
#process.AODSIMEventContent.outputCommands.append('keep *_distortedMuons_*_*')
#process.myEventContent = process.AODSIMEventContent
process.myEventContent = cms.PSet(
  outputCommands = cms.untracked.vstring(
      #'keep *_distortedMuons_*_*',
      'keep *'
  )
)

process.wmnOutput = cms.OutputModule("PoolOutputModule",
      process.myEventContent,
      SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring('distortMuons')
      ),
      fileName = cms.untracked.string('root_files/distortedMuons.root')
)

# Output histograms
#process.TFileService = cms.Service("TFileService", fileName = cms.string('WMuNu_histograms.root') )

# Steering the process
process.distortMuons = cms.Path(process.genMatchMap*process.distortedMuons*process.wmnSelFilter)
process.end = cms.EndPath(process.wmnOutput)
