import FWCore.ParameterSet.Config as cms

min_iov = 340000
max_iov = 999999999
subSystemName = "TotemTiming"

process = cms.Process('test')

# some of the printouts in PrintTotemDAQMapping are done on INFO level
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    )
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(min_iov),
    lastValue = cms.uint64(min_iov),
    interval = cms.uint64(1)
)

# load a mapping from XML
process.load("CalibPPS.ESProducers.totemDAQMappingESSourceXML_cfi")
process.totemDAQMappingESSourceXML.subSystem = subSystemName
process.totemDAQMappingESSourceXML.sampicSubDetId = cms.uint32(5)
process.totemDAQMappingESSourceXML.configuration = cms.VPSet(
  cms.PSet(
    validityRange = cms.EventRange(f"{min_iov}:min - {max_iov}:max"),
    mappingFileNames = cms.vstring('CondFormats/PPSObjects/xml/mapping_totem_timing_2022.xml'),
    maskFileNames = cms.vstring(),
  )
)

# load a mapping from DB
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = "sqlite_file:CTPPSTotemTiming_DAQMapping.db"
process.PoolDBESSource = cms.ESSource('PoolDBESSource',
    process.CondDB,
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('TotemReadoutRcd'),
            tag = cms.string('DiamondDAQMapping'),
            label = cms.untracked.string(subSystemName)
        ),
        cms.PSet(
            record = cms.string('TotemAnalysisMaskRcd'),
            tag = cms.string('AnalysisMask'),
            label = cms.untracked.string(subSystemName)
        )
    )
)

# prefer to read mapping from DB than from XML or otherwise
process.es_prefer_totemTimingMapping = cms.ESPrefer("PoolDBESSource", "", TotemReadoutRcd=cms.vstring(f"TotemDAQMapping/TotemTiming"))

# print the mapping
process.writeTotemDAQMapping = cms.EDAnalyzer("WriteTotemDAQMapping",
  subSystem = cms.untracked.string(subSystemName),
  fileName = cms.untracked.string("all_timing_db.txt")
)

process.path = cms.Path(
  process.writeTotemDAQMapping
)