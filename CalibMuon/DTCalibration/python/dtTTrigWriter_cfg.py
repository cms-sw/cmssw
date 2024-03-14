import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("PROD",eras.Run3)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run3_data']

process.load('Configuration/StandardSequences/GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.DTGeometryESModule.applyAlignment = False
process.DTGeometryESModule.fromDDD = False

process.load("CondCore.CondDB.CondDB_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# tried following
# https://cmssdt.cern.ch/SDT/doxygen/CMSSW_7_4_1/doc/html/d3/d48/popcon2dropbox__job__conf_8py_source.html
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    logconnect = cms.string('sqlite_file:ttrig.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    ))
)

process.load("CalibMuon.DTCalibration.dtTTrigWriter_cfi")
process.dtTTrigWriter.kFactor = -0.7
process.dtTTrigWriter.rootFileName = 'DTTimeBoxes.root'
process.dtTTrigWriter.debug = False

process.p = cms.Path(process.dtTTrigWriter)
# dummy dummy
# dummy dummy
# dummy dummy
# dummy dummy
# dummy dummy
