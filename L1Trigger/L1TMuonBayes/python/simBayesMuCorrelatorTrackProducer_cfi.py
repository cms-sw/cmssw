import FWCore.ParameterSet.Config as cms

simBayesMuCorrelatorTrackProducer = cms.EDProducer("L1TMuonBayesMuCorrelatorTrackProducer",
                              
  srcDTPh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcDTTh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcCSC = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
  srcRPC = cms.InputTag('simMuonRPCDigis'), 
  g4SimTrackSrc = cms.InputTag('g4SimHits'),                             
  dumpResultToXML = cms.bool(False),
  dumpDetailedResultToXML = cms.bool(False),
  XMLDumpFileName = cms.string("TestEvents.xml"),                                     
  dumpGPToXML = cms.bool(False),  
  readEventsFromXML = cms.bool(False),
  eventsXMLFiles = cms.vstring("TestEvents.xml"),
  dropRPCPrimitives = cms.bool(False),                                    
  dropDTPrimitives = cms.bool(False),                                    
  dropCSCPrimitives = cms.bool(False),
  processorType = cms.string("MuCorrelatorProcessor"),
  #ttTracksSource = cms.string("L1_TRACKER"),
  ttTracksSource = cms.string("SIM_TRACKS"), 
               L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),               ## TTTrack input
               MCTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"), ## MCTruth input 
               # other input collections
               L1StubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
               MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
               MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
               TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
               TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
               l1Tk_nPar = cms.int32(4),         # use 4 or 5-parameter L1 track fit ??
               l1Tk_minNStub = cms.int32(4),     # L1 tracks with >= 4 stubs
          
  pdfModuleFileName = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/pdfModuleSimTracks100FilesSigma1p3.xml"),                                     
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/Patterns_0x0003_TT_withBitShiftAndHighDtQual.xml"),
  #pdfModuleFileName = cms.FileInPath("L1Trigger/L1TMuonBayes/test/pdfModule.xml"),
  #pdfModuleType = cms.string("PdfModuleWithStats")
  #refLayerMustBeValid = cms.bool(True),

  #bxRangeMin = cms.int32(-10),
  #bxRangeMax = cms.int32(10),
  
  useStubsFromAdditionalBxs = cms.int32(0), #default is 0, if 1 it allows trigger on HSPCs
)

