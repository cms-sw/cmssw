import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8PhotonFluxSettings_cfi import PhotonFlux_PbPb

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(5360.),
    PhotonFlux = PhotonFlux_PbPb,
    PythiaParameters = cms.PSet(
        pythia8_example02 = cms.vstring('HardQCD:all = on',
                                        'PhaseSpace:pTHatMin = 10.',
                                        'PhotonParton:all = on',#Added from main70
                                        'MultipartonInteractions:pT0Ref = 3.0',#Added from main70
                                        'PDF:beamA2gamma = on',#Added from main70
                                        'PDF:proton2gammaSet = 0',#Added from main70
                                        'PDF:useHardNPDFB = on',
                                        'PDF:gammaFluxApprox2bMin = 13.272',
                                        'PDF:beam2gammaApprox = 2',
                                        'Photon:sampleQ2 = off'
                                    ), 
        parameterSets = cms.vstring('pythia8_example02')
    )
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(2)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('pythia8ex2.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
