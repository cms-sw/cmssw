import FWCore.ParameterSet.Config as cms

TkAlCaRecoMonitor = cms.EDAnalyzer("TkAlCaRecoMonitor",
                                   TrackProducer= cms.InputTag( "generalTracks" ),
                                   CaloJetCollection= cms.InputTag( "kt6CaloJets" ),
                                   AlgoName = cms.string("testTkAlCaReco"),
                                   MeasurementState = cms.string("default"),#All/OuterSurface/InnerSurface/ImpactPoint/default(track)                         
                                   runsOnReco = cms.bool(True),
                                   fillInvariantMass = cms.bool(False),
                                   #
                                   maxJetPt = cms.double(10), #GeV
                                   #
                                   MassBin = cms.uint32(100),
                                   MassMin = cms.double(0.0),
                                   MassMax = cms.double(100.0),
                                   #
                                   JetPtBin= cms.uint32(100),
                                   JetPtMin = cms.double(0.0),
                                   JetPtMax = cms.double(50.0),
                                   #
                                   MinJetDeltaRBin=cms.uint32(100),
                                   MinJetDeltaRMin = cms.double(0),
                                   MinJetDeltaRMax = cms.double(10),
                                   #
                                   MinTrackDeltaRBin=cms.uint32(100),
                                   MinTrackDeltaRMin = cms.double(0),
                                   MinTrackDeltaRMax = cms.double(3.2),
                                   #
                                   FolderName = cms.string("TkAlCaRecoMonitor"),
                                   OutputMEsInRootFile = cms.bool(False),
                                   OutputFileName = cms.string("MonitorTrack.root")
                                   )
                                   
