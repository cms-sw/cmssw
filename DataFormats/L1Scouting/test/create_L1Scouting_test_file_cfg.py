import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0],
    description='Test writing EDM products with L1Scouting data formats',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-n', '--maxEvents', type=int, default=1,
    help='Value of process.maxEvents.input')

parser.add_argument('-o', '--outputFileName', type=str, default='testL1Scouting.root',
    help='Output file name')

parser.add_argument('-s', '--splitLevel', type=int, default=99,
    help='Split level of ROOT branches in EDM output file')

args = parser.parse_args()

process = cms.Process("PROD")

process.source = cms.Source("EmptySource")
process.maxEvents.input = args.maxEvents

process.l1ScoutingTestProducer = cms.EDProducer("TestWriteL1Scouting",
    bxValues = cms.vuint32(42, 512),
    muonValues = cms.vint32(1, 2, 3),
    jetValues = cms.vint32(4, 5, 6, 7),
    eGammaValues = cms.vint32(8, 9, 10),
    tauValues = cms.vint32(11, 12),
    bxSumsValues = cms.vint32(13),
    bmtfStubValues = cms.vint32(1, 2),
    caloTowerValues = cms.vint32(14, 15, 16, 17, 18),
    fastJetFloatingPointValues = cms.vdouble(19., 20., 21.),
    fastJetIntegralValues = cms.vint32(22, 23, 24)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(args.outputFileName),
    splitLevel = cms.untracked.int32(args.splitLevel)
)

process.path = cms.Path(process.l1ScoutingTestProducer)

process.endPath = cms.EndPath(process.out)
