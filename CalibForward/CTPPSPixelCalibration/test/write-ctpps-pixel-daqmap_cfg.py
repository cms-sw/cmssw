import FWCore.ParameterSet.Config as cms

process = cms.Process('test')


process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

#process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(1)
#)

# load a mapping
process.load("CondFormats.CTPPSReadoutObjects.CTPPSPixelDAQMappingESSourceXML_cfi")

#Database output service
process.load("CondCore.CondDB.CondDB_cfi")
# output database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:CTPPSPixel_DAQMapping_AnalysisMask.db'


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
    cms.PSet(
        record = cms.string('CTPPSPixelDAQMappingRcd'),
        tag = cms.string('PixelDAQMapping'),
        label = cms.string('RPix')
    )
  )
)


# print the mapping and analysis mask
process.writeCTPPSPixelDAQMapping = cms.EDAnalyzer("WriteCTPPSPixelDAQMapping",
    cms.PSet(
        daqmappingiov = cms.uint64(1),
        record = cms.string("CTPPSPixelDAQMappingRcd"),
        label = cms.string("RPix")
    )
)

process.path = cms.Path(
  process.writeCTPPSPixelDAQMapping
)
