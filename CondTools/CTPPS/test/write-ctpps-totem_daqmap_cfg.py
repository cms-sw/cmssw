import FWCore.ParameterSet.Config as cms

process = cms.Process('writeTotemDAQMappingMask')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.MessageLogger = cms.Service("MessageLogger",
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
      threshold = cms.untracked.string('INFO'),
      
  ))

# load a mapping from XML file, set dummy validity range
process.load("CalibPPS.ESProducers.totemDAQMappingESSourceXML_cfi")
process.totemDAQMappingESSourceXML.subSystem = "TimingDiamond"
process.totemDAQMappingESSourceXML.sampicSubDetId = cms.uint32(6)
process.totemDAQMappingESSourceXML.configuration = cms.VPSet(
  cms.PSet(
    validityRange = cms.EventRange("1:min - 362919:max"),
    mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2022.xml"),
    maskFileNames = cms.vstring()
  )
)

#Database output service
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:CTPPSDiamond_DAQMapping.db'
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
    cms.PSet(
        record = cms.string('TotemReadoutRcd'),
        tag = cms.string('DiamondDAQMapping'),
        label = cms.string('TimingDiamond')
    ),
    cms.PSet(
        record = cms.string('TotemAnalysisMaskRcd'),
        tag = cms.string('AnalysisMask'),
        label = cms.string('TimingDiamond')
    )
    
  )
)

# print the mapping and write it to DB
process.writeCTPPSTotemDAQMappingMask = cms.EDAnalyzer("WriteCTPPSTotemDAQMappingMask",
    cms.PSet(
        daqmappingiov = cms.uint64(1),
        record_map = cms.string('TotemReadoutRcd'),
        record_mask = cms.string('TotemAnalysisMaskRcd'),
        label = cms.string("TimingDiamond"),
    )
)

process.content = cms.EDAnalyzer("EventContentAnalyzer") 

process.path = cms.Path(
  process.writeCTPPSTotemDAQMappingMask
)