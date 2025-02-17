import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Generate XML geometry.')
parser.add_argument("--geom", help="Name of parameter", type=str, default='Extended2023')
parser.add_argument("--out", help="Prefix for output file", type=str, default='ge')

args = parser.parse_args()
import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryXMLWriter")

process.load("Configuration.Geometry.Geometry"+args.geom+"_cff")

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.BigXMLWriter = cms.EDAnalyzer("OutputDDToDDL",
                                      rotNumSeed = cms.int32(0),
                                      fileName = cms.untracked.string("./"+args.out+"SingleBigFile.xml")
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.BigXMLWriter)

