import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.autoCond import autoCond

process = cms.Process("GeometryTest")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = autoCond['run1_mc']

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(3),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("PHGCalRcd"),
             tag = cms.string("HGCALRECO_Geometry_Test01"),
             connect = cms.string("sqlite_file:myfile.db")
             )
    )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        HGCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.prodEE = cms.EDAnalyzer("HGCalGeometryTester",
                                Detector   = cms.string("HGCalEESensitive"),
                                SquareCell = cms.bool(True),
                                )

process.prodHEF = process.prodEE.clone(
    Detector   = "HGCalHESiliconSensitive",
    SquareCell = True
)

process.prodHEB = process.prodEE.clone(
    Detector   = "HGCalHEScintillatorSensitive",
    SquareCell = True
)

process.p1 = cms.Path(process.prodEE+process.prodHEF+process.prodHEB)
