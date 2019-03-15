import FWCore.ParameterSet.Config as cms

###OMTF emulator configuration
simOmtfDigis = cms.EDProducer("L1TMuonBayesOmtfTrackProducer",
                              
  srcDTPh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcDTTh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcCSC = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
  srcRPC = cms.InputTag('simMuonRPCDigis'), 
  g4SimTrackSrc = cms.InputTag('g4SimHits'),                             
  dumpResultToXML = cms.bool(True),
  dumpDetailedResultToXML = cms.bool(False),
  XMLDumpFileName = cms.string("TestEvents.xml"),                                     
  dumpGPToXML = cms.bool(False),  
  readEventsFromXML = cms.bool(False),
  eventsXMLFiles = cms.vstring("TestEvents.xml"),
  dropRPCPrimitives = cms.bool(False),                                    
  dropDTPrimitives = cms.bool(False),                                    
  dropCSCPrimitives = cms.bool(False),
  processorType = cms.string("OMTFProcessor"),
  ttTracksSource = cms.string("SIM_TRACKS"), 
  ghostBusterType = cms.string("GhostBusterPreferRefDt"),
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/optimisedPats0.xml"),
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x00020007.xml")
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0003.xml")   
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/GPs_ArtWithThresh.xml")
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/GPs78_withThresh2.xml")
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/GPsNorm1NoCor_WithThresh4.xml")                             

  #  bxMin = cms.int32(-3),
  #  bxMax = cms.int32(4)
)

