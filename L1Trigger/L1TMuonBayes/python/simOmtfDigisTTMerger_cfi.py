import FWCore.ParameterSet.Config as cms

###OMTF emulator configuration
simOmtfDigis = cms.EDProducer("L1TMuonBayesOmtfTTMergerTrackProducer",
                              
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
  processorType = cms.string("OMTFProcessorTTMerger"),
  ttTracksSource = cms.string("L1_TRACKER"),
  
               L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),               ## TTTrack input
               MCTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"), ## MCTruth input 
               # other input collections
               L1StubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
               MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
               MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
               TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
               TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
               l1Tk_nPar = cms.int32(4),         # use 4 or 5-parameter L1 track fit ??
                                               
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/Patterns_0x0003_TT_withBitShiftAndHighDtQual.xml"),
   patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0003_TT_withBitShiftAndHighDtQual.xml"),
  
  refLayerMustBeValid = cms.bool(True),
  #ghostBusterType = cms.string("GhostBusterPreferRefDt"),
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/optimisedPats0.xml"),
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x00020007.xml")
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0003.xml")   
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/GPs_ArtWithThresh.xml")
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/GPs78_withThresh2.xml")
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/GPsNorm1NoCor_WithThresh4.xml")     
  

  #  bxMin = cms.int32(-3),
  #  bxMax = cms.int32(4)
)

