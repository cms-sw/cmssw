import FWCore.ParameterSet.Config as cms

minIov = 368023
maxIov = 999999999
subSystemName = "TotemT2"


process = cms.Process('writeTotemDAQMappingMask')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(minIov),
    lastValue = cms.uint64(minIov),
    interval = cms.uint64(1)
)

process.MessageLogger = cms.Service("MessageLogger",
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
      threshold = cms.untracked.string('ERROR'),    
  )
)

# load a mapping from XML file, set dummy validity range
process.load("CalibPPS.ESProducers.totemDAQMappingESSourceXML_cfi")
process.totemDAQMappingESSourceXML.subSystem = subSystemName
process.totemDAQMappingESSourceXML.sampicSubDetId = cms.uint32(7)
process.totemDAQMappingESSourceXML.multipleChannelsPerPayload = True
process.totemDAQMappingESSourceXML.configuration = cms.VPSet(
  cms.PSet(
    validityRange = cms.EventRange(f"{minIov}:min - {maxIov}:max"),
    mappingFileNames = cms.vstring('CondFormats/PPSObjects/xml/mapping_totem_nt2_2023_final.xml'),
    maskFileNames = cms.vstring(),
  )
)

#Database output service
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = "sqlite_file:CTPPSTotemT2_DAQMapping.db"
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
        daqMappingIov = cms.uint64(minIov),
        recordMap = cms.string('TotemReadoutRcd'),
        recordMask = cms.string('TotemAnalysisMaskRcd'),
        label = cms.string(subSystemName),
    )
)

process.content = cms.EDAnalyzer("EventContentAnalyzer") 

process.path = cms.Path(
  process.writeCTPPSTotemDAQMappingMask
)
