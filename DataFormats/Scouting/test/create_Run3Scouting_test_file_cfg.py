import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.run3ScoutingProducer = cms.EDProducer("TestWriteRun3Scouting",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values. Note only values exactly convertible to
    # float are used to avoid precision and rounding issues in
    # in comparisons.
    caloJetsValues = cms.vdouble(
        2.0,   4.0 , 6.0,  8.0, 10.0,
        12.0, 14.0, 16.0, 18.0, 20.0,
        22.0, 24.0, 26.0, 28.0, 30.0,
        32.0
    ),
    electronsFloatingPointValues = cms.vdouble(
        10.0,   20.0,  30.0,  40.0,  50.0,
        60.0,   70.0,  80.0,  90.0, 100.0,
        110.0, 120.0, 130.0, 140.0, 150.0,
        160.0, 170.0, 180.0, 190.0, 200.0,
        210.0, 220.0, 230.0, 240.0, 250.0,
        260.0, 270.0, 280.0, 290.0, 300.0,
        310.0, 320.0, 330.0
    ),
    electronsIntegralValues = cms.vint32(
        10, 20, 30, 40, 50, 60, 70, 80
    ),
    muonsFloatingPointValues = cms.vdouble(
        10.0,   20.0,  30.0,  40.0,  50.0,
        60.0,   70.0,  80.0,  90.0, 100.0,
        110.0, 120.0, 130.0, 140.0, 150.0,
        160.0, 170.0, 180.0, 190.0, 200.0,
        210.0, 220.0, 230.0, 240.0, 250.0,
        260.0, 270.0, 280.0, 290.0, 300.0,
        310.0, 320.0, 330.0, 340.0, 350.0,
        360.0, 370.0
    ),
    muonsIntegralValues = cms.vint32(
        10,   20,  30,  40,  50,
        60,   70,  80,  90, 100,
        110, 120, 130, 140, 150,
        160, 170, 180, 190, 200,
        210, 220, 230, 240, 250,
        260
    ),
    particlesFloatingPointValues = cms.vdouble(
        11.0,   21.0,  31.0,  41.0,  51.0,
        61.0,   71.0,  81.0,  91.0, 101.0,
        111.0
    ),
    particlesIntegralValues = cms.vint32(
        11,   21,  31,  41,  51
    ),
    pfJetsFloatingPointValues = cms.vdouble(
        12.0,   22.0,  32.0,  42.0,  52.0,
        62.0,   72.0,  82.0,  92.0, 102.0,
        112.0, 122.0, 132.0, 142.0, 152.0
    ),
    pfJetsIntegralValues = cms.vint32(
        12,   22,  32,  42,  52,
        62,   72,  82
    ),
    photonsFloatingPointValues = cms.vdouble(
        14.0,   23.0,  33.0,  43.0,  53.0,
        63.0,   73.0,  83.0,  93.0, 103.0,
        113.0, 123.0, 133.0, 143.0, 153.0,
        163.0, 173.0
    ),
    photonsIntegralValues = cms.vint32(
        14,   23,  33, 43, 53
    ),
    tracksFloatingPointValues = cms.vdouble(
        14.0,   24.0,  34.0,  44.0,  54.0,
        64.0,   74.0,  84.0,  94.0, 104.0,
        114.0, 124.0, 134.0, 144.0, 154.0,
        164.0, 174.0, 184.0, 194.0, 204.0,
        214.0, 224.0, 234.0, 244.0, 254.0,
        264.0, 274.0, 284.0, 294.0
    ),
    tracksIntegralValues = cms.vint32(
        14,  24,  34,  44,  54
    ),
    vertexesFloatingPointValues = cms.vdouble(
        15.0,   25.0,  35.0,  45.0,  55.0,
        65.0,   75.0,  85.0,  95.0,  105.0
    ),
    vertexesIntegralValues = cms.vint32(
        15,  25,  35
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRun3Scouting.root')
)

process.path = cms.Path(process.run3ScoutingProducer)
process.endPath = cms.EndPath(process.out)
