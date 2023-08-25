import FWCore.ParameterSet.Config as cms

minIov = 368023
maxIov = 999999999
subSystemName = "TotemT2"

process = cms.Process('test')

# some of the printouts in PrintTotemDAQMapping are done on INFO level
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    )
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(minIov),
    lastValue = cms.uint64(minIov),
    interval = cms.uint64(1)
)

# load a mapping from XML
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

# load a mapping from DB
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = "sqlite_file:CTPPSTotemT2_DAQMapping.db"
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
process.es_prefer_totemTimingMapping = cms.ESPrefer("TotemDAQMappingESSourceXML",                 "totemDAQMappingESSourceXML", TotemReadoutRcd=cms.vstring("TotemDAQMapping/TotemT2"))

# print the mapping
process.writeTotemDAQMapping = cms.EDAnalyzer("WriteTotemDAQMapping",
  subSystem = cms.untracked.string(subSystemName),
  fileName = cms.untracked.string("all_TotemT2_xml.txt")
)

process.path = cms.Path(
  process.writeTotemDAQMapping
)