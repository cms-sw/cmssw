import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Generate XML geometry.')
parser.add_argument("--geom", help="Name of parameter", type=str, default='ExtendedGeometry2026x')
parser.add_argument("--out", help="Prefix for output file", type=str, default='ge')

args = parser.parse_args()

process = cms.Process("GeometryXMLWriter")

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/CMSCommonData/data/dd4hep/cms'+args.geom+'.xml'),
                                            appendToDataLabel = cms.string('make-payload')
                                           )

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                  appendToDataLabel = cms.string('make-payload')
                                                )

process.BigXMLWriter = cms.EDAnalyzer("OutputDD4hepToDDL",
                              fileName = cms.untracked.string("./"+args.out+"SingleBigFile.xml")
                              )


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.BigXMLWriter)

