#Using evtgen plugin

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.),
    useEvtGenPlugin = cms.PSet(),
    #evtgenDecFile = cms.string("mydecfile"),
    #evtgenPdlFile = cms.string("mypdlfile"),
    evtgenUserFile = cms.vstring('GeneratorInterface/Pythia8Interface/test/evtgen_userfile.dec'),
    PythiaParameters = cms.PSet(
        pythia8_example11 = cms.vstring('HardQCD:hardbbbar = on'),
        parameterSets = cms.vstring('pythia8_example11')
    )
)

# in order to use lhapdf PDF add a line like this to pythia8_example02:
# 'PDF:pSet = LHAPDF5:MRST2004nlo.LHpdf'

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
    fileName = cms.untracked.string('pythia8ex11.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
