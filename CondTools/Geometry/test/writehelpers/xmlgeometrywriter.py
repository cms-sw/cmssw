import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Generate XML geometry.')
parser.add_argument("--tag", help="global tag to use", type=str)
parser.add_argument("--out", help="part of out file name", type=str, default='Extended')
parser.add_argument("--inPre", help="Prefix for input geometry file", type=str, default='ge')

args = parser.parse_args()

process = cms.Process("XMLGeometryWriter")

process.load('CondCore.CondDB.CondDB_cfi')

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.XMLGeometryWriter = cms.EDAnalyzer("XMLGeometryBuilder",
                                           XMLFileName = cms.untracked.string("./"+args.inPre+"SingleBigFile.xml"),
                                           ZIP = cms.untracked.bool(True)
                                           )

process.CondDB.timetype = cms.untracked.string('runnumber')
process.CondDB.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'),tag = cms.string('XMLFILE_Geometry_'+args.tag+'_'+args.out+'_mc')))
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.XMLGeometryWriter)
