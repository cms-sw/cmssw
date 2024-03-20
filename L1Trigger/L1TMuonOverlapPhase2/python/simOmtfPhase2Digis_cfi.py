import FWCore.ParameterSet.Config as cms


###OMTF emulator configuration
simOmtfPhase2Digis = cms.EDProducer("L1TMuonOverlapPhase2TrackProducer",
                              
  srcDTPh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcDTTh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcCSC = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
  srcRPC = cms.InputTag('simMuonRPCDigis'), 
  srcDTPhPhase2 = cms.InputTag('dtTriggerPhase2PrimitiveDigis'),
  
  simTracksTag = cms.InputTag('g4SimHits'),                             
  dumpResultToXML = cms.bool(False),
  dumpDetailedResultToXML = cms.bool(False),
  XMLDumpFileName = cms.string("TestEvents.xml"),                                     
  dumpGPToXML = cms.bool(False),  
  readEventsFromXML = cms.bool(False),
  eventsXMLFiles = cms.vstring("TestEvents.xml"),
  
  
  dropRPCPrimitives = cms.bool(False),                                    
  dropCSCPrimitives = cms.bool(False),
  
  dropDTPrimitives = cms.bool(True),  
  usePhase2DTPrimitives = cms.bool(True), #if usePhase2DTPrimitives is True,  dropDTPrimitives must be True as well
  
  processorType = cms.string("OMTFProcessor"),
  
  #if commented the default values are 0-0
  #-3 to 4 is the range of the OMTF DAQ readout, so should be used e.g. in the DQM data to emulator comparison
  bxMin = cms.int32(0),
  bxMax = cms.int32(0),
  
  noHitValueInPdf = cms.bool(True),
  minDtPhiQuality = cms.int32(2),
  minDtPhiBQuality = cms.int32(4),
  
  dtRefHitMinQuality =  cms.int32(4),
  
  dtPhiBUnitsRad = cms.int32(1024), #2048 is the orginal phase2 scale, 512 is the phase1 scale
    
  stubEtaEncoding = cms.string("valueP1Scale"), #TODO change to valueP1Scale when InputMakerPhase2 is modifiwed
  
  usePhiBExtrapolationFromMB1 = cms.bool(False),
  usePhiBExtrapolationFromMB2 = cms.bool(False),
  useStubQualInExtr  = cms.bool(False),
  useEndcapStubsRInExtr  = cms.bool(False),
  useFloatingPointExtrapolation  = cms.bool(False),
  
  sorterType = cms.string("byLLH"),
  ghostBusterType = cms.string("GhostBusterPreferRefDt"), # byLLH byRefLayer GhostBusterPreferRefDt
  goldenPatternResultFinalizeFunction = cms.int32(9)
)
