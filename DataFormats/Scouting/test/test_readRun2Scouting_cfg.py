import FWCore.ParameterSet.Config as cms
import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test Run 2 Scouting data formats')

parser.add_argument("--muonVersion", type=int, help="muon data format version (default: 3)", default=3)
parser.add_argument("--trackVersion", type=int, help="track data format version (default: 2)", default=2)
parser.add_argument("--vertexVersion", type=int, help="vertex data format version (default: 3)", default=3)
parser.add_argument("--inputFile", type=str, help="Input file name (default: testRun2Scouting.root)", default="testRun2Scouting.root")
parser.add_argument("--outputFileName", type=str, help="Output file name (default: testRun2Scouting2.root)", default="testRun2Scouting2.root")
argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+args.inputFile))

process.testReadRun2Scouting = cms.EDAnalyzer("TestReadRun2Scouting",
    # I stick to values exactly convertable to float
    # to avoid potential rounding issues in the test.
    expectedCaloJetsValues = cms.vdouble(
        2.0,   4.0 , 6.0,  8.0, 10.0,
        12.0, 14.0, 16.0, 18.0, 20.0,
        22.0, 24.0, 26.0, 28.0, 30.0,
        32.0
    ),
    caloJetsTag = cms.InputTag("run2ScoutingProducer", "", "PROD"),
    expectedElectronFloatingPointValues = cms.vdouble(
        10.0,   20.0,  30.0,  40.0,  50.0,
        60.0,   70.0,  80.0,  90.0, 100.0,
        110.0, 120.0, 130.0, 140.0
    ),
    expectedElectronIntegralValues = cms.vint32(10, 20),
    electronsTag = cms.InputTag("run2ScoutingProducer", "", "PROD"),
    muonClassVersion = cms.int32(args.muonVersion),
    expectedMuonFloatingPointValues = cms.vdouble(
        10.0,   20.0,  30.0,  40.0,  50.0,
        60.0,   70.0,  80.0,  90.0, 100.0,
        110.0, 120.0, 130.0, 140.0, 150.0,
        160.0, 170.0, 180.0, 190.0, 200.0,
        210.0, 220.0, 230.0
    ),
    expectedMuonIntegralValues = cms.vint32(
        10,   20,  30,  40,  50,
        60,   70,  80
    ),
    muonsTag = cms.InputTag("run2ScoutingProducer", "", "PROD"),
    expectedParticleFloatingPointValues = cms.vdouble(
        11.0,   21.0,  31.0,  41.0
    ),
    expectedParticleIntegralValues = cms.vint32(
        11,   21
    ),
    particlesTag = cms.InputTag("run2ScoutingProducer", "", "PROD"),
    expectedPFJetFloatingPointValues = cms.vdouble(
        12.0,   22.0,  32.0,  42.0,  52.0,
        62.0,   72.0,  82.0,  92.0, 102.0,
        112.0, 122.0, 132.0, 142.0, 152.0
    ),
    expectedPFJetIntegralValues = cms.vint32(
        12,   22,  32,  42,  52,
        62,   72,  82
    ),
    pfJetsTag = cms.InputTag("run2ScoutingProducer", "", "PROD"),
    expectedPhotonFloatingPointValues = cms.vdouble(
        14.0,   23.0,  33.0,  43.0,  53.0,
        63.0,   73.0,  83.0
    ),
    photonsTag = cms.InputTag("run2ScoutingProducer", "", "PROD"),
    trackClassVersion = cms.int32(args.trackVersion),
    expectedTrackFloatingPointValues = cms.vdouble(
        215.0,   225.0,  235.0,  245.0,  255.0,
        265.0,   275.0,  285.0,  295.0,  305.0,
        315.0,   325.0,  335.0,  345.0,  355.0,
        365.0
    ),
    expectedTrackIntegralValues = cms.vint32(
        52,   62,  72,  82
    ),
    tracksTag = cms.InputTag("run2ScoutingProducer", "", "PROD"),
    vertexClassVersion = cms.int32(args.vertexVersion),
    expectedVertexFloatingPointValues = cms.vdouble(
        15.0,   25.0,  35.0,  45.0,  55.0,
        65.0,   75.0
    ),
    expectedVertexIntegralValues = cms.vint32(
        12,   22,  32
    ),
    vertexesTag = cms.InputTag("run2ScoutingProducer", "", "PROD")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(args.outputFileName)
)

process.path = cms.Path(process.testReadRun2Scouting)

process.endPath = cms.EndPath(process.out)
