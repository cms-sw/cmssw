import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('GR09_P_V8_34X::All')
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    #'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/A81DC948-A2E6-DE11-AF25-000423D94524.root',
    '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec14thReReco_v1/0100/965030FF-3DE9-DE11-B75B-00151796C18C.root',
    '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec14thReReco_v1/0100/485EDF6C-48E9-DE11-BE46-0024E876841F.root',
    '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec14thReReco_v1/0100/3EAFDE66-71E9-DE11-B861-0024E876A814.root',
    '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec14thReReco_v1/0099/A683E072-22E9-DE11-B396-001D0967DA6C.root',
    '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec14thReReco_v1/0099/5E275AD5-35E9-DE11-96BA-00151796D884.root',
    '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec14thReReco_v1/0099/04E1641C-2AE9-DE11-A073-001D0967CFCC.root'
  ),
  #skipEvents = cms.untracked.uint32(0)
  #eventsToProcess = cms.untracked.VEventRange('124009:10872958-124009:10872958')
  eventsToProcess = cms.untracked.VEventRange('124120:542515-124120:542515')
)

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
process.load("JetMETCorrections.Configuration.ZSPJetCorrections332_cff")
process.JetPlusTrackZSPCorrectorAntiKt5.Verbose = True
import JetMETCorrections.JetPlusTrack.JPTTrackProducer_cfi
process.jet1Tracks = JetMETCorrections.JetPlusTrack.JPTTrackProducer_cfi.JPTTrackProducer.clone()
process.jet1Tracks.JetIndex = 0
process.jet1Tracks.ZSPCorrectedJetsTag = 'ZSPJetCorJetAntiKt5'
process.jet1Tracks.JPTCorrectorName = 'JetPlusTrackZSPCorrectorAntiKt5'
process.jet2Tracks = JetMETCorrections.JetPlusTrack.JPTTrackProducer_cfi.JPTTrackProducer.clone()
process.jet2Tracks.JetIndex = 1
process.jet2Tracks.ZSPCorrectedJetsTag = 'ZSPJetCorJetAntiKt5'
process.jet2Tracks.JPTCorrectorName = 'JetPlusTrackZSPCorrectorAntiKt5'
process.jptTracksForFireworks = cms.Sequence( process.jet1Tracks * process.jet2Tracks )

process.dump = cms.EDAnalyzer('EventContentAnalyzer')

process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("jptTracks.root")
)

#process.p1 = cms.Path( process.dump )
process.p1 = cms.Path( process.ZSPJetCorrectionsAntiKt5 * process.JetPlusTrackCorrectionsAntiKt5 * process.jptTracksForFireworks )
process.e1 = cms.EndPath( process.output )
