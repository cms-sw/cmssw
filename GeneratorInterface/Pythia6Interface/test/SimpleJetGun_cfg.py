import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.Config as cms

process = cms.Process("Gen")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("Pythia6JetGun",
    
    maxEventsToPrint = cms.untracked.int32(5),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(True),    
       
    # Generate a number of particles with some fermi motion
    PGunParameters = cms.PSet(
        ParticleID = cms.vint32(211,-211,111,111,130),
        # this defines "absolute" energy spead of particles in the jet
	MinE   = cms.double(0.5),
	MaxE   = cms.double(2.0),
	# the following params define the boost
        MinP   = cms.double(20.0),
        MaxP   = cms.double(20.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
	MinEta = cms.double(-2.4),
        MaxEta = cms.double(2.4)
    ),
    
    # no detailed pythia6 settings necessary    
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    
    )

)

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('SimpleJet.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p,process.outpath)
