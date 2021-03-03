import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerMapProd")


process.MessageLogger = cms.Service("MessageLogger",
    cablingReader = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring(''),
    files = cms.untracked.PSet(
        cablingMap = cms.untracked.PSet(

        )
    )
)

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
process.siStripCond.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripFedCablingRcd'),
    tag = cms.string('SiStripFedCabling_Fake_30X')
))
process.siStripCond.connect = 'sqlite_file:dbfile.db'
process.siStripCond.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.load("DQMServices.Core.DQM_cfg")

process.siStripCablingTrackerMap = cms.EDAnalyzer("SiStripCablingTrackerMap")

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(50908),
    lastValue = cms.uint64(50908),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)



process.p = cms.Path(process.siStripCablingTrackerMap)
