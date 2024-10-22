import FWCore.ParameterSet.Config as cms

###OMTF emulator configuration
simOmtfDigis = cms.EDProducer("L1TMuonOverlapPhase1TrackProducer",

  srcDTPh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcDTTh = cms.InputTag('simDtTriggerPrimitiveDigis'),
  srcCSC = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
  srcRPC = cms.InputTag('simMuonRPCDigis'),

  #g4SimTrackSrc = cms.InputTag('g4SimHits'),
  dumpResultToXML = cms.bool(False),
  dumpDetailedResultToXML = cms.bool(False),
  XMLDumpFileName = cms.string("TestEvents.xml"),
  dumpGPToXML = cms.bool(False),
  readEventsFromXML = cms.bool(False),
  eventsXMLFiles = cms.vstring("TestEvents.xml"),
  dropRPCPrimitives = cms.bool(False),
  dropDTPrimitives = cms.bool(False),
  dropCSCPrimitives = cms.bool(False),
  processorType = cms.string("OMTFProcessor"),

  #ghostBusterType = cms.string("GhostBusterPreferRefDt"),

  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x00020007.xml")
  #patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0003.xml")

  #if commented the default values are 0-0
  #-3 to 4 is the range of the OMTF DAQ readout, so should be used e.g. in the DQM data to emulator comparison
  bxMin = cms.int32(0),
  bxMax = cms.int32(0),

  extrapolFactorsFilename = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/ExtrapolationFactors_simple.xml"),
)

### Era: Run3_2024
from Configuration.Eras.Modifier_stage2L1Trigger_2024_cff import stage2L1Trigger_2024 
stage2L1Trigger_2024.toModify(simOmtfDigis, 
                              configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0009.xml"),
                              patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_ExtraplMB1nadMB2SimplifiedFP_t17_classProb17_recalib2_minDP0_v3.xml") )
