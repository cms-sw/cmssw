import FWCore.ParameterSet.Config as cms

###OMTF emulator configuration
simOmtfPhase2Digis = cms.EDProducer("L1TMuonOverlapPhase2TrackProducer",                         
  srcDTPh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcDTTh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcCSC = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
  #srcCSC = cms.InputTag('simCscTriggerPrimitiveDigis'),
  srcRPC = cms.InputTag('simMuonRPCDigis'), 
  srcDTPhPhase2 = cms.InputTag('dtTriggerPhase2PrimitiveDigis'),
  srcDTThPhase2 = cms.InputTag('dtTriggerPhase2PrimitiveDigis'),

  ##  XML / PATTERNS file:
  configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0209.xml"),
  extrapolFactorsFilename = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/ExtrapolationFactors_ExtraplMB1nadMB2_R_EtaValueP1Scale_t35.xml"),
  patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_ExtraplMB1andMB2RFixedP_ValueP1Scale_DT_2_2_2_t35__classProb17_recalib2.xml"),

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
  minDtPhiBQuality = cms.int32(2),
  
  dtRefHitMinQuality =  cms.int32(4),
  
  dtPhiBUnitsRad = cms.int32(1024), #2048 is the orginal phase2 scale, 512 is the phase1 scale
    
  stubEtaEncoding = cms.string("valueP1Scale"), #TODO change to valueP1Scale when InputMakerPhase2 is modifiwed
  
  rpcMaxClusterSize = cms.int32(3),
  rpcMaxClusterCnt = cms.int32(2),
  rpcDropAllClustersIfMoreThanMax = cms.bool(True),

  usePhiBExtrapolationFromMB1 = cms.bool(True),
  usePhiBExtrapolationFromMB2 = cms.bool(True),
  #in the DTTriggerPhase2 in PR44924 the phi is defined always in the middle of the chamber, even for the uncorelated stubs
  #so the qulaity doeas not maater in the extrapolation
  useStubQualInExtr  = cms.bool(False),
  useEndcapStubsRInExtr  = cms.bool(True),
  useFloatingPointExtrapolation  = cms.bool(False),

  sorterType = cms.string("byLLH"),
  ghostBusterType = cms.string("byRefLayer"), # byLLH byRefLayer GhostBusterPreferRefDt
  goldenPatternResultFinalizeFunction = cms.int32(10)
)
