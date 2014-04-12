import FWCore.ParameterSet.Config as cms

#
# WARNING: as of 12/1/2010, this is NOT an example for users -
#          it's my private (JY) "development" cfg, for testing
#          tauola++ package - which is NOT yet added to CMSSW -
#          and the corresponding updated TauolaInterface
#


process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    comEnergy = cms.double(7000.),
    
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
	     UseTauolaPolarization = cms.bool(True),
	     InputCards = cms.PSet
	     ( 
	        # 0 = generic decay
		pjak1 = cms.int32(0), # 1 = electron mode
		pjak2 = cms.int32(0), # 2 = muon mode
		mdtau = cms.int32(0)  # does NOT matter at this point...
	     )
	),
        parameterSets = cms.vstring('Tauola')
    ),
    
    PythiaParameters = cms.PSet(
        py8GeneralSettings = cms.vstring(  'Main:timesAllowErrors    = 10000', 
                                           'ParticleDecays:limitTau0 = on',
	                                   'ParticleDecays:tauMax = 10'
        ),
        py8UESettings = cms.vstring(       'Tune:pp 2',
	                                   'Tune:ee 3'
	),
        py8SpecialSettings   = cms.vstring('HadronLevel:Hadronize = off',     # mstp(111)=1
				           #' ParticleDecays:limitTau0 = on',  # mstj(22)=2 - decay unstable particles
				           # 'ParticleDecays:tau0Max   = 10.', # parj(71)=10.- for which ctau < 10 mm
					   'PartonLevel:ISR = off',
					   'PartonLevel:FSR = off',
					   'PartonLevel:MI = off'           # mstp(81)=0
					   # for pp intractions, SpaceShower things are not important,
					   # but TimeShower settings are relevat
					   'TimeShower:QEDshowerByL = off',
					   'TimeShower:QEDshowerByQ = off',
					   'SpaceShower:QEDshowerByL = off',
					   'SpaceShower:QEDshowerByQ = off',
					   'PhaseSpace:pTHatMin = 5.',
					   'PhaseSpace:pTHatMax = 150.',
					   'StringZ:usePetersonB = on', # these 2 together ==
					   'StringZ:usePetersonC = on'  # mstj(11)=3
	),
        py8ProcessSettings = cms.vstring(  # 'HiggsSM:gg2H = on',
					   # 'HiggsSM:ff2Hff(t:ZZ) = on',
					   # 'HiggsSM:ff2Hff(t:WW) = on'
					   # 'WeakDoubleBoson:ffbar2ZW = on' 
					   # 'WeakSingleBoson:ffbar2gmZ = on' # msel=11
					   'WeakSingleBoson:ffbar2W = on' 
	),
	py8WDecaySettings = cms.vstring(   '24:onMode = off', 
	                                   '24:onIfAny = 15' 
	),
        parameterSets = cms.vstring('py8GeneralSettings',
	                            'py8UESettings',
	                            'py8ProcessSettings',
				    'py8WDecaySettings' 
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
process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.Timing=cms.Service("Timing",
    summaryOnly=cms.untracked.bool(True))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Test_Py8_W2TauNu_TauolaCxx.root')
)

process.p = cms.Path(process.generator)
process.p1 = cms.Path(process.randomEngineStateProducer)
process.outpath = cms.EndPath(process.GEN)

#process.schedule = cms.Schedule(process.p, process.p1, process.outpath)
process.schedule = cms.Schedule(process.p)

