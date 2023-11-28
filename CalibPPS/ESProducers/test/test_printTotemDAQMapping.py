import FWCore.ParameterSet.Config as cms

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
    firstValue = cms.uint64(340000),
    lastValue = cms.uint64(340000),
    interval = cms.uint64(1)
)

# load a mapping from XML
process.load("CalibPPS.ESProducers.totemDAQMappingESSourceXML_cfi")
process.totemDAQMappingESSourceXML.subSystem = "TimingDiamond"
process.totemDAQMappingESSourceXML.sampicSubDetId = 6
process.totemDAQMappingESSourceXML.configuration = cms.VPSet(
  cms.PSet(
    validityRange = cms.EventRange("340000:min - 362925:max"),
    mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2022.xml"),
    maskFileNames = cms.vstring()
  )
)

# load a mapping from DB
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = 'sqlite_file:CTPPSDiamondsScript_DAQMapping.db' # SQLite input
process.PoolDBESSource = cms.ESSource('PoolDBESSource',
    process.CondDB,
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('TotemReadoutRcd'),
            tag = cms.string('DiamondDAQMapping'),
            label = cms.untracked.string('TimingDiamond')
        ),
        cms.PSet(
            record = cms.string('TotemAnalysisMaskRcd'),
            tag = cms.string('AnalysisMask'),
            label = cms.untracked.string('TimingDiamond')
        )
    )
)

# prefer to read mapping from DB than from XML
# use this line to read from DB
process.es_prefer_totemTimingMapping = cms.ESPrefer("PoolDBESSource","", TotemReadoutRcd=cms.vstring("TotemDAQMapping/TimingDiamond"))
# use this line to read from XML
#process.es_prefer_totemTimingMapping = cms.ESPrefer("TotemDAQMappingESSourceXML","totemDAQMappingESSourceXML",TotemReadoutRcd=cms.vstring("TotemDAQMapping/TimingDiamond"))

# print the mapping
process.printTotemDAQMapping = cms.EDAnalyzer("PrintTotemDAQMapping",
  subSystem = cms.untracked.string("TimingDiamond")
)

process.path = cms.Path(
  process.printTotemDAQMapping
)