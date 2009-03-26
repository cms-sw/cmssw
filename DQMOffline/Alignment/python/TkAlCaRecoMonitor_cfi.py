import FWCore.ParameterSet.Config as cms

TkAlCaRecoMonitor = cms.EDAnalyzer("TkAlCaRecoMonitor",
                                   TrackProducer= cms.InputTag( "generalTracks" ),
                                   ReferenceTrackProducer= cms.InputTag( "generalTracks" ),
                                   CaloJetCollection= cms.InputTag( "kt6CaloJets" ),
                                   AlgoName = cms.string("testTkAlCaReco"),
                                   MeasurementState = cms.string("default"),#All/OuterSurface/InnerSurface/ImpactPoint/default(track)                         
                                   runsOnReco = cms.bool(False),
                                   fillInvariantMass = cms.bool(False),
                                   useSignedR = cms.bool(False),
                                   #
                                   TrackEfficiencyBin =cms.uint32(102),
                                   TrackEfficiencyMin = cms.double(-0.01),
                                   TrackEfficiencyMax = cms.double(1.01),
                                   #
                                   maxJetPt = cms.double(10), #GeV
                                   #
                                   SumChargeBin = cms.uint32(11),
                                   SumChargeMin = cms.double(-5.5),
                                   SumChargeMax = cms.double(5.5),
                                   #
                                   MassBin = cms.uint32(100),
                                   MassMin = cms.double(0.0),
                                   MassMax = cms.double(100.0),
                                   #
                                   TrackPtBin= cms.uint32(110),
                                   TrackPtMin = cms.double(0.0),
                                   TrackPtMax = cms.double(110.0),
                                   #
                                   TrackCurvatureBin= cms.uint32(2000),
                                   TrackCurvatureMin = cms.double(-0.01),
                                   TrackCurvatureMax = cms.double(0.01),
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
                                   HitMapsZBin = cms.uint32(300),
                                   HitMapZMax = cms.double(300.), #cm
                                   HitMapsRBin = cms.uint32(120),
                                   HitMapRMax = cms.double(120.), #cm
                                   #
                                   daughterMass = cms.double(0.10565836),#Gev
                                   FolderName = cms.string("TkAlCaRecoMonitor"),
                                   OutputMEsInRootFile = cms.bool(False),
                                   OutputFileName = cms.string("TkAlCaRecoDQM.root")
                                   )
                                   
