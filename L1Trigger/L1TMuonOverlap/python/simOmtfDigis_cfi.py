import FWCore.ParameterSet.Config as cms

###OMTF emulator configuration
simOmtfDigis = cms.EDProducer("L1TMuonOverlapTrackProducer",
                              srcDTPh = cms.InputTag('simDtTriggerPrimitiveDigis'),
                              srcDTTh = cms.InputTag('simDtTriggerPrimitiveDigis'),
                              srcCSC = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
                              srcRPC = cms.InputTag('simMuonRPCDigis'),                              
                              dumpResultToXML = cms.bool(False),
                              dumpDetailedResultToXML = cms.bool(False),
                              XMLDumpFileName = cms.string("TestEvents.xml"),                                     
                              dumpGPToXML = cms.bool(False),  
                              readEventsFromXML = cms.bool(False),
                              eventsXMLFiles = cms.vstring("TestEvents.xml"),
                              dropRPCPrimitives = cms.bool(False),                                    
                              dropDTPrimitives = cms.bool(False),                                    
                              dropCSCPrimitives = cms.bool(False),   
                              omtf = cms.PSet(
                                  configFromXML = cms.bool(False),   
                                  patternsXMLFiles = cms.VPSet(
                                       cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x00020007.xml")),
                                      ),
                                  configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x00020005.xml"),
                              )
)


