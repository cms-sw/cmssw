import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
#process.maxEvents.input = 10
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.run2ScoutingProducer = cms.EDProducer("TestWriteRun2Scouting",
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
        110.0, 120.0, 130.0, 140.0
    ),
    electronsIntegralValues = cms.vint32(
        10, 20
    ),
    muonsFloatingPointValues = cms.vdouble(
        10.0,   20.0,  30.0,  40.0,  50.0,
        60.0,   70.0,  80.0,  90.0, 100.0,
        110.0, 120.0, 130.0, 140.0, 150.0,
        160.0, 170.0, 180.0, 190.0, 200.0,
        210.0, 220.0, 230.0
    ),
    muonsIntegralValues = cms.vint32(
        10,   20,  30,  40,  50,
        60,   70,  80
    ),
    particlesFloatingPointValues = cms.vdouble(
        11.0,   21.0,  31.0,  41.0
    ),
    particlesIntegralValues = cms.vint32(
        11,   21
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
        63.0,   73.0,  83.0
    ),
    tracksFloatingPointValues = cms.vdouble(
        215.0,   225.0,  235.0,  245.0,  255.0,
        265.0,   275.0,  285.0,  295.0,  305.0,
        315.0,   325.0,  335.0,  345.0,  355.0,
        365.0
    ),
    tracksIntegralValues = cms.vint32(
        52,   62,  72,  82
    ),
    vertexesFloatingPointValues = cms.vdouble(
        15.0,   25.0,  35.0,  45.0,  55.0,
        65.0,   75.0
    ),
    vertexesIntegralValues = cms.vint32(
        12,   22,  32
    ) 
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRun2Scouting.root')
)

process.path = cms.Path(process.run2ScoutingProducer)
process.endPath = cms.EndPath(process.out)
