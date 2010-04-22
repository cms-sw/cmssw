import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(10),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    comEnergy = cms.double(7000.),
    
    ExternalDecays = cms.PSet(
        Photos = cms.untracked.PSet(),
        parameterSets = cms.vstring( "Photos" )
    ),
    
    PythiaParameters = cms.PSet(
        py8SpecialSettings   = cms.vstring( 'HadronLevel:Hadronize = off', # mstp(111)=0
				            'ParticleDecays:limitTau0 = on',  # mstj(22)=2 - decay unstable particles
				            'ParticleDecays:tau0Max   = 10.', # parj(71)=10.- for which ctau < 10 mm
	                                    'PartonLevel:ISR = off',       # mstp(61)=0
	                                    'PartonLevel:FSR = off',       # mstp(71)=0 ; what about mstj(41) ???
					    'PartonLevel:MI = off'         # mstp(81)=0
					    # for pp intractions, SpaceShower things are not important,
					    # but TimeShower settings are relevat
					    'TimeShower:QEDshowerByL = off',
					    'TimeShower:QEDshowerByQ = off'
	),
        py8ProcessSettings = cms.vstring( 'StringZ:usePetersonB = on', # these 2 together ==
	                                  'StringZ:usePetersonC = on', # mstj(11)=3
					  'HiggsSM:gg2H = on',         # mstp(102)=1
					  'HiggsSM:ff2Hff(t:ZZ) = on', # msub(123)=1 - ZZ fusion to H
					  'HiggsSM:ff2Hff(t:WW) = on', # msub(124)=1 - WW fusion to H
					  '25:m0 = 190.0',             # pmas(25,1)=190.0 - mass of H 
	                                  'PhaseSpace:pTHatMin = 5.',  # ckin(45)=5.
	                                  'PhaseSpace:pTHatMax = 150.' # ckin(46)=150.
					  # what about choice of structure function ??? (mstp(51)=7)
	),
	py8HiggsDecaySettings = cms.vstring('25:onMode = off', # turn OFF all Higgs decays
	                                    '25:onIfAny = 23'  # now turn ON H->ZZ
	),
	py8ZDecaySettings = cms.vstring(  '23:onMode = off', # turn OFF all Z decays
	                                  '23:onIfAny = 13', # turn ON Z->mumu
					  '23:onIfAny = 15'  # turn ON Z->tautau
	),
	py8TauDecaySettings = cms.vstring('15:onMode = off', # turn OFF all tau decays
	                                  '15:onIfAny = 11', # turn ON tau -> e
					  '15:onIfAny = 13'  # turn ON tau -> mu
	),
        parameterSets = cms.vstring('py8SpecialSettings','py8ProcessSettings',
	                            'py8HiggsDecaySettings',
				    'py8ZDecaySettings','py8TauDecaySettings')
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
    fileName = cms.untracked.string('Py8ZHH_Photos.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)

