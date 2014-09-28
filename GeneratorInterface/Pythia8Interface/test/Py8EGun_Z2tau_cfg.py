import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8EGun",

    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(True),

    PGunParameters = cms.PSet(
       ParticleID = cms.vint32(23),
       AddAntiParticle = cms.bool(True),
       MinPhi = cms.double(-3.14159265359),
       MaxPhi = cms.double(3.14159265359),
       MinE = cms.double(100.0),
       MaxE = cms.double(200.0),
       MinEta = cms.double(0.0),
       MaxEta = cms.double(2.4)
    ),
        
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
	     UseTauolaPolarization = cms.bool(True),
	     InputCards = cms.PSet
	     ( 
	        pjak1 = cms.int32(0), # 1 = electron mode
		pjak2 = cms.int32(0), # 2 = muon mode
		mdtau = cms.int32(240)  # (any) tau -> nu pi+- 
	     )
	),
        parameterSets = cms.vstring('Tauola')
    ),

    PythiaParameters = cms.PSet(
	py8ZDecaySettings = cms.vstring(  '23:onMode = off', # turn OFF all Z decays
					  '23:onIfAny = 15'  # turn ON Z->tautau
	),
        parameterSets = cms.vstring(  
	                              'py8ZDecaySettings' 
				   )
    )
)

# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []
process.MessageLogger.fwkJobReports = []

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Py8EGun_Z2tautau.root')
)


process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)

