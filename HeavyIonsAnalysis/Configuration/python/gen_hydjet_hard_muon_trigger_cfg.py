import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")
process.load("GeneratorInterface.PyquenInterface.pyquenPythiaDefault_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('HydjetSource'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(0)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.untracked.uint32(17974)
)

process.source = cms.Source("HydjetSource",
    shadowingSwitch = cms.int32(0),
    maxTransverseRapidity = cms.double(1.0),
    comEnergy = cms.double(5500.0),
    sigmaInelNN = cms.double(58.0),
    doRadiativeEnLoss = cms.bool(True),
    qgpInitialTemperature = cms.double(1.0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    aBeamTarget = cms.double(207.0),
    cFlag = cms.int32(0),
    hydjetMode = cms.string('kHydroQJets'),
    hadronFreezoutTemperature = cms.double(0.14),
    nMultiplicity = cms.int32(26000),
    qgpNumQuarkFlavor = cms.int32(0),
    doCollisionalEnLoss = cms.bool(True),
    bFixed = cms.double(0.0),
    maxLongitudinalRapidity = cms.double(3.75),
    bMin = cms.double(0.0),
    fracSoftMultiplicity = cms.double(1.0),
    maxEventsToPrint = cms.untracked.int32(0),
    bMax = cms.double(0.0),
    PythiaParameters = cms.PSet(
        process.pyquenPythiaDefaultBlock,
        parameterSets = cms.vstring('pythiaMuons')
    ),
    qgpProperTimeFormation = cms.double(0.1)
)

process.gentrig = cms.EDFilter("GenTrigger",
    printResults = cms.untracked.bool(True),
    etaMin = cms.untracked.double(-2.4),
    trigIDs = cms.vint32(13),
    etaMax = cms.untracked.double(2.4),
    ptMin = cms.untracked.double(20.0)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('hydjet_muontrig_gen_x4_c20_d20080529.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('trig')
    )
)

process.trig = cms.Path(process.gentrig)
process.outpath = cms.EndPath(process.out)
process.HydjetSource.bMax = 7.21126
process.HydjetSource.bMin = 0
process.HydjetSource.cFlag = 20
process.HydjetSource.hydjetMode = 'kQJetsOnly'


