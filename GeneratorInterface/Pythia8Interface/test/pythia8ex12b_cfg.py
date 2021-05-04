import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.),

   #Turning on CMSSW Pythia8Interface resonance scale setting and pythia8 emission veto:

    PythiaParameters = cms.PSet(
        pythia8PowhegEmissionVetoSettings = cms.vstring(
              'POWHEG:veto = 1',
              'POWHEG:pTdef = 1',
              'POWHEG:emitted = 0',
              'POWHEG:pTemt = 0',
              'POWHEG:pThard = 0',
              'POWHEG:vetoCount = 100',
              'SpaceShower:pTmaxMatch = 2',
              'TimeShower:pTmaxMatch = 2',
        ),
        pythia8_example12 = cms.vstring(
              'POWHEG:bb4l = on',
              'Beams:frameType = 4',
              'Beams:LHEF = powheg-b_bbar_4l.lhe',
        ),
        parameterSets = cms.vstring('pythia8PowhegEmissionVetoSettings', 'pythia8_example12')
    )

)

#if emissionVeto is on && MPIveto > 0 add 'MultipartonInteractions:pTmaxMatch = 2'

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
    input = cms.untracked.int32(100)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('pythia8ex12.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
