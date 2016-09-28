import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.),
    PythiaParameters = cms.PSet(
        pythia8_pthook = cms.vstring(
            'SoftQCD:nonDiffractive = on',         # QCD process, all quark are produced, but bquark (5) only 1.4% of the time, 
                                                   # Lets hadronize just those, 
#
            'PTFilter:filter = on',                # turn on the filter, for testing turn off and see how increase the number of 
                                                   # required pythia events to pass process.bfilter
            'PTFilter:quarkToFilter = 5',          # filter in b quark
            'PTFilter:scaleToFilter = 1.0',        # at the scale shawering of 1 GeV (this should be not affect the kinematical distribution at low pT)
            'PTFilter:quarkRapidity = 10.',        # do nothing on the rapidity of this quark, (at the most quark are about 8 units)
            'PTFilter:quarkPt = -0.1'              # do nothing of the pT of the quark
        ),
        parameterSets = cms.vstring('pythia8_pthook')
    )
)

process.bfilter = cms.EDFilter("PythiaFilter",ParticleID = cms.untracked.int32(5))

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(2)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(initialSeed = cms.untracked.uint32(123456789))
)

#when no PTFilter is off. will need this amount to have about 150 in the output
#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10000))

#when PTFilter is on, just those
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(147))

process.GEN = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string('pthookfilter.root'),
        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p'))
)

process.p = cms.Path(process.generator*process.bfilter)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
