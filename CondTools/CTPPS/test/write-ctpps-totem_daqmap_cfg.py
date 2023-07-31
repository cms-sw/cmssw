import FWCore.ParameterSet.Config as cms

min_iov = 362920
max_iov = 999999999
subSystemName = "TimingDiamond"


process = cms.Process('writeTotemDAQMappingMask')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(min_iov),
    lastValue = cms.uint64(min_iov),
    interval = cms.uint64(1)
)

process.MessageLogger = cms.Service("MessageLogger",
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
      threshold = cms.untracked.string('INFO'),
      
  ))

# load a mapping from XML file, set dummy validity range
process.load("CalibPPS.ESProducers.totemDAQMappingESSourceXML_cfi")
process.totemDAQMappingESSourceXML.subSystem = subSystemName
process.totemDAQMappingESSourceXML.sampicSubDetId = cms.uint32(6)
process.totemDAQMappingESSourceXML.configuration = cms.VPSet(
  cms.PSet(
    validityRange = cms.EventRange(f"{min_iov}:min - {max_iov}:max"),
    mappingFileNames = cms.vstring('CondFormats/PPSObjects/xml/mapping_timing_diamond_2023.xml'),
    maskFileNames = cms.vstring(),
  )
)

#Database output service
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = "sqlite_file:CTPPSDiamondsScript_DAQMapping.db"
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
    cms.PSet(
        record = cms.string('TotemReadoutRcd'),
        tag = cms.string('DiamondDAQMapping'),
        label = cms.string(subSystemName)
    ),
    cms.PSet(
        record = cms.string('TotemAnalysisMaskRcd'),
        tag = cms.string('AnalysisMask'),
        label = cms.string(subSystemName)
    )
    
  )
)

# print the mapping and write it to DB
process.writeCTPPSTotemDAQMappingMask = cms.EDAnalyzer("WriteCTPPSTotemDAQMappingMask",
    cms.PSet(
        daqmappingiov = cms.uint64(min_iov),
        record_map = cms.string('TotemReadoutRcd'),
        record_mask = cms.string('TotemAnalysisMaskRcd'),
        label = cms.string(subSystemName),
    )
)

process.content = cms.EDAnalyzer("EventContentAnalyzer") 

process.path = cms.Path(
  process.writeCTPPSTotemDAQMappingMask
)
