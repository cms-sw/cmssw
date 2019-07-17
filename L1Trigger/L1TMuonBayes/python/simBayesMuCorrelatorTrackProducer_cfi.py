import FWCore.ParameterSet.Config as cms

simBayesMuCorrelatorTrackProducer = cms.EDProducer("L1TMuonBayesMuCorrelatorTrackProducer",
                              
  srcDTPh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcDTTh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcCSC = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
  srcRPC = cms.InputTag('simMuonRPCDigis'), 
  dropRPCPrimitives = cms.bool(False),                                    
  dropDTPrimitives = cms.bool(False),                                    
  dropCSCPrimitives = cms.bool(False),
  processorType = cms.string("MuCorrelatorProcessor"),
  ttTracksSource = cms.string("L1_TRACKER"), #"TRACKING_PARTICLES"
  L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),  ## TTTrack input
  #g4SimTrackSrc = cms.InputTag('g4SimHits'), 
  #MCTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"), ## MCTruth input 
  # other input collections
  #L1StubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
  #MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
  #MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
  #TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
  #TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
  
  l1Tk_nPar = cms.int32(4),         # use 4 or 5-parameter L1 track fit ??
  l1Tk_minNStub = cms.int32(4),     # L1 tracks with >= 4 stubs
  pdfModuleFile = cms.FileInPath("L1Trigger/L1TMuon/data/muonBayesCorrelator_config/muCorrelatorPdfModule.xml"), 
  timingModuleFile  = cms.FileInPath("L1Trigger/L1TMuon/data/muonBayesCorrelator_config/muCorrelatorTimingModule.xml"),
  #pdfModuleType = cms.string("PdfModuleWithStats")

  #bxRangeMin = cms.int32(-10),
  #bxRangeMax = cms.int32(10),
  
  useStubsFromAdditionalBxs = cms.int32(3), #if 0 then only the muon stubs from the same BX as the ttTrack (i.e. BX=0) are used for the correlation, if >=1 then also stubs from selcted number of the BX are used. It allows trigger on HSPCs
)

