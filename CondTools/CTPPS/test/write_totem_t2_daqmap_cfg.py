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
process.load("CalibPPS.ESProducers.totemT2DAQMapping_cff")

#Database output service
process.load("CondCore.CondDB.CondDB_cfi")
# output database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:TotemT2_DAQMapping.db'


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
    cms.PSet(
        record = cms.string('TotemT2DAQMappingRcd'),
        tag = cms.string('T2DAQMapping'),
        label = cms.string('T2')
    )
  )
)


# print the mapping and analysis mask
process.writeTotemT2DAQMapping = cms.EDAnalyzer("WriteTotemT2DAQMapping",
    cms.PSet(
        daqmappingiov = cms.uint64(1),
        record = cms.string("TotemT2DAQMappingRcd"),
        label = cms.string("T2")
    )
)

process.path = cms.Path(
  process.writeTotemT2DAQMapping
)
