# Validate and benchmark HGCalFastGeometry on any Phase-2 geometry: the
# HGCalGeometry::getPosition() of every sampled cell must map back to the same
# DetId through HGCalFastGeometry::toDetId(). One job covers all requested
# subdetectors, so the geometry is built once.
#
# Usage:
#   cmsRun hgcalFastGeometryTester_cfg.py --geom D110 --era Phase2C17I13M9
#   cmsRun hgcalFastGeometryTester_cfg.py --geom D122 --era Phase2C26I13M9 \
#       --sense HGCalHESiliconSensitive
# The unit test (test/BuildFile.xml) runs D122 with --maxCells 20000 --benchmark 0.
import argparse
import functools
import importlib
import sys

import FWCore.ParameterSet.Config as cms

allSenses = ["HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"]

parser = argparse.ArgumentParser()
parser.add_argument("--geom", default="D110", help="GeometryExtendedRun4<geom>Reco_cff")
parser.add_argument("--era", default="Phase2C17I13M9", help="era matching the geometry")
parser.add_argument("--sense", default="all",
                    help="all, or a comma-separated list of " + ", ".join(allSenses))
parser.add_argument("--maxCells", type=int, default=200000,
                    help="cells per subdetector, sampled over the whole detector; -1 = all")
parser.add_argument("--benchmark", type=int, default=1, help="1 = also time toDetId() against getClosestCell()")
args = parser.parse_args(sys.argv[2:] if sys.argv[0].endswith("cmsRun") else sys.argv[1:])

era = getattr(importlib.import_module(f"Configuration.Eras.Era_{args.era}_cff"), args.era)
process = cms.Process("GEOTEST", era)
process.load(f"Configuration.Geometry.GeometryExtendedRun4{args.geom}Reco_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.HGCalFastGeom = cms.untracked.PSet(limit=cms.untracked.int32(-1))

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1))

testers = []
for sense in (allSenses if args.sense == "all" else args.sense.split(",")):
    tester = cms.EDAnalyzer("HGCalFastGeometryTester",
                            nameSense=cms.string(sense),
                            benchmark=cms.bool(bool(args.benchmark)),
                            maxCells=cms.int32(args.maxCells))
    setattr(process, "test" + sense, tester)
    testers.append(tester)
process.p = cms.Path(functools.reduce(lambda a, b: a + b, testers))
