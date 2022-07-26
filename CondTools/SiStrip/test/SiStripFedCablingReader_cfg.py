import FWCore.ParameterSet.Config as cms

process = cms.Process("FedCablingReader")

process.MessageLogger = cms.Service("MessageLogger",
    threshold = cms.untracked.string('INFO'),
    debugModules = cms.untracked.vstring(''),
    #destinations = cms.untracked.vstring('cout'),
    destinations = cms.untracked.vstring('cablingReader.log')
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:SiStripConditionsDBFile.db'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripFedCablingRcd'),
        tag = cms.string('SiStripFedCabling_30X')
    ))
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase1_2022_design']
print("taking geometry from %s" % process.GlobalTag.globaltag.value())
process.load("Configuration.StandardSequences.GeometryDB_cff")

process.SiStripConnectivity = cms.ESProducer("SiStripConnectivity")
process.SiStripRegionConnectivity = cms.ESProducer("SiStripRegionConnectivity",
                                                   EtaDivisions = cms.untracked.uint32(20),
                                                   PhiDivisions = cms.untracked.uint32(20),
                                                   EtaMax = cms.untracked.double(2.5)
)

process.prefer_SiStripConnectivity = cms.ESPrefer("SiStripConnectivity", "sistripconn")
process.prefer_SiStripCabling = cms.ESPrefer("PoolDBESSource", "poolDBESSource")

process.fedcablingreader = cms.EDAnalyzer("SiStripFedCablingReader",
                                          PrintFecCabling = cms.untracked.bool(True),
                                          PrintDetCabling = cms.untracked.bool(True),
                                          PrintRegionCabling = cms.untracked.bool(True)
)

process.p1 = cms.Path(process.fedcablingreader)


