import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[2]))

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
    expectedMuonFloatingPointValues = cms.vdouble(
        10.0,   20.0,  30.0,  40.0,  50.0,
        60.0,   70.0,  80.0,  90.0, 100.0,
        110.0
    ),
    expectedMuonIntegralValues = cms.vint32(
        10,   20,  30,  40,  50,
        60
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
    expectedVertexFloatingPointValues = cms.vdouble(
        15.0,   25.0,  35.0,  45.0
    ),
    vertexesTag = cms.InputTag("run2ScoutingProducer", "", "PROD")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRun2Scouting2.root')
)

process.path = cms.Path(process.testReadRun2Scouting)

process.endPath = cms.EndPath(process.out)
