import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('GR09_P_V8_34X::All')
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/F80561F5-A2E6-DE11-B342-000423D996C8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/F4767383-9FE6-DE11-8099-000423D985B0.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/EE646B4A-99E6-DE11-AF39-001D09F2AF96.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/E826F9B2-AAE6-DE11-94AB-003048D3750A.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/E466542D-A8E6-DE11-89BB-000423D9853C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/DEF9A66D-ABE6-DE11-9F1D-000423D33970.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/DCCA559F-9CE6-DE11-97E4-000423D98804.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/DC5C34E5-B3E6-DE11-9526-003048D2BE08.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/D67048AB-A3E6-DE11-9AB0-000423D94990.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/BC780CBB-A5E6-DE11-AF91-001D09F2932B.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/B2C169DD-A0E6-DE11-80FE-001D09F2525D.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/AA8AF135-A0E6-DE11-9C48-001D09F2426D.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/A81DC948-A2E6-DE11-AF25-000423D94524.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/A459C2E4-A7E6-DE11-925D-001D09F252E9.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/A286ECF8-A2E6-DE11-9C21-001617C3B778.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/A256FE95-A1E6-DE11-9F5D-000423D98B6C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/98271EC4-AAE6-DE11-A34B-003048D37514.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/8EEB3E47-A2E6-DE11-8D31-000423D951D4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/86A0E2DE-A0E6-DE11-945E-001D09F2462D.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/80E7934F-9DE6-DE11-B5F3-001D09F25041.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/808DAB04-9EE6-DE11-A45D-0019B9F704D6.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/6AA2CC3F-A7E6-DE11-883A-001D09F295FB.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/62A1C6E9-A7E6-DE11-B136-0019B9F730D2.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/36FB7AF0-9BE6-DE11-BED4-001D09F28755.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/32F48C17-A5E6-DE11-ABCC-001617DBD224.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/282EFE2C-A8E6-DE11-AFFD-000423D990CC.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/24F54541-A7E6-DE11-A90F-000423D99AA2.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/0ECDEC97-A1E6-DE11-BB90-000423D986C4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/0E502584-9FE6-DE11-B954-000423D9989E.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/009/0CF4E657-A9E6-DE11-9021-0030486730C6.root'
  ),
  #skipEvents = cms.untracked.uint32(0)
  eventsToProcess = cms.untracked.VEventRange('124009:10872958-124009:10872958')
)

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
process.load("JetMETCorrections.Configuration.ZSPJetCorrections332_cff")
process.JetPlusTrackZSPCorrectorAntiKt7.Verbose = True
import JetMETCorrections.JetPlusTrack.JPTTrackProducer_cfi
process.jet1Tracks = JetMETCorrections.JetPlusTrack.JPTTrackProducer_cfi.JPTTrackProducer.clone()
process.jet1Tracks.JetIndex = 0
process.jet2Tracks = JetMETCorrections.JetPlusTrack.JPTTrackProducer_cfi.JPTTrackProducer.clone()
process.jet2Tracks.JetIndex = 1
process.jptTracksForFireworks = cms.Sequence( process.jet1Tracks * process.jet2Tracks )

process.dump = cms.EDAnalyzer('EventContentAnalyzer')

process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("jptTracks.root")
)

#process.p1 = cms.Path( process.dump )
process.p1 = cms.Path( process.ZSPJetCorrectionsAntiKt7 * process.jptTracksForFireworks )
#process.p1 = cms.Path( process.RawToDigi * process.reconstruction * process.ZSPJetCorrectionsAntiKt7 * process.jptTracksForFireworks )
process.e1 = cms.EndPath( process.output )
