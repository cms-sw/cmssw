import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("EmptySource",
     firstLuminosityBlock = cms.untracked.uint32(1),
     numberEventsInLuminosityBlock = cms.untracked.uint32(200),
     firstEvent = cms.untracked.uint32(1),
     firstRun = cms.untracked.uint32(1),
     numberEventsInRun = cms.untracked.uint32(1000)

)

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    comEnergy = cms.double(10000.),
        
    PythiaParameters = cms.PSet(
        py8UESettings = cms.vstring( 'StringZ:usePetersonB = on',      # these 2 together ==
	                             'StringZ:usePetersonC = on',      # mstj(11)=3
				     'ParticleDecays:limitTau0 = on',  # mstj(22)=2 - decay unstable particles
				     'ParticleDecays:tau0Max   = 10.', # parj(71)=10.- for which ctau < 10 mm
				     # 'PartonLevel:ISR = on',         # on==D; mstp(61)=1 or 2 ??? Py6 D=2
	                             # 'PartonLevel:FSR = on',         # on==D;  mstp(71)=1 
				     'SigmaProcess:alphaSorder=1',     # D; mstp(2)=1 (D)
				     'SigmaProcess:Kfactor=1.',        # D; mstp(33)=0 - no K-factor, i.e. K=1
				     # This is the important master switch - I can turn it ON/OFF, 
				     # and it'll greatly affect the event multiplicity, 
				     # whithout messing up too much the principle physics
				     'HadronLevel:Hadronize = on',
				     'PartonLevel:MI = on',            # D; mstp(81)=1
				     ### Py6 mstp(82)=4 (MI model) does NOT have anything similar in Py8, 
				     ### because it actually means a new model, and in Py8 old models went away 
				     ### - only new remain
				     'MultipleInteractions:pT0Ref=1.8387',    # parp(82)=1.8387 - pt cutoff for MI
				     'MultipleInteractions:ecmRef=1960.',     # parp(89)=1960.
				     'MultipleInteractions:coreFraction=0.5', # D; parp(83)=0.5
				     'MultipleInteractions:coreRadius=0.4',   # D; parp(84)=0.4
				     ### parp(85) & parp(86) are N/A in Py8
				     'MultipleInteractions:ecmPow=0.16',      # parp(90)=0.16
				     'BeamRemnants:primordialKT=on',          # D; mstp(91)=1
				     'BeamRemnants:primordialKThard=2.1',     # parp(91)=2.1
				     'BeamRemnants:primordialKTremnant=15.',  # parp(93)=15.
				     'SpaceShower:pTmin=1.25',                # parp(62)=1.25
				     ### parp(64) is N/A in Py8
				     'SpaceShower:pTmaxFudge=2.'              # should this be instead of parp(67)=2.5 ??? 
				                                              # In Py6, D==parp(67)=4.
				                                              # and in Py8 for this D=1, max=2
	),
        py8SpecialSettings   = cms.vstring( 'HadronLevel:Hadronize = off',     # mstp(111)=1
				          #'ParticleDecays:limitTau0 = on',  # mstj(22)=2 - decay unstable particles
				          #'ParticleDecays:tau0Max   = 10.', # parj(71)=10.- for which ctau < 10 mm
					  'PartonLevel:ISR = off',
					  'PartonLevel:FSR = off',
					  'PartonLevel:MI = on'           # mstp(81)=0
					  # for pp intractions, SpaceShower things are not important,
					  # but TimeShower settings are relevat
					  'TimeShower:QEDshowerByL = off',
					  'TimeShower:QEDshowerByQ = off',
					  'SpaceShower:QEDshowerByL = off',
					  'SpaceShower:QEDshowerByQ = off',
					  'PhaseSpace:pTHatMin = 5.',
					  'PhaseSpace:pTHatMax = 150.'
	),
        py8ProcessSettings = cms.vstring( 'StringZ:usePetersonB = on', # these 2 together ==
	                                  'StringZ:usePetersonC = on', # mstj(11)=3
					  'WeakSingleBoson:ffbar2gmZ = on' # msel=11
					  # what about choice of structure function ??? (mstp(51)=7)
	),
	py8ZDecaySettings = cms.vstring(  '23:onMode = off', # turn OFF all Z decays
	                                  #'23:onIfAny = 13', # turn ON Z->mumu
					  '23:onIfAny = 15'  # turn ON Z->tautau
					  #'23:onIfAny = 5'
	),
	py8TauDecaySettings = cms.vstring('15:onMode = off', # turn OFF all tau decays
	                                  '15:onIfAny = 11', # turn ON tau -> e
					  '15:onIfAny = 13'  # turn ON tau -> mu
	),
        parameterSets = cms.vstring(   'py8UESettings',
	                            #'py8SpecialSettings',
	                            'py8ProcessSettings',
				    'py8ZDecaySettings')
    )
)

# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
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
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Py8_Z2tautau.root')
)

#
# needed for the BasicGenTester (analyzer)
#
#process.TFileService = cms.Service("TFileService",
#        fileName = cms.string("BasicGenTester_Py8_Ztautau.root")
#)
# the analyzer itself  
#
#process.BasicGenTest = cms.EDAnalyzer( "BasicGenTester",
#        NPartForHisto = cms.untracked.int32(1000),
#        PtMaxForHisto = cms.untracked.double(5.0)
#)
#process.p1 = cms.Path( process.BasicGenTest )

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
#process.schedule = cms.Schedule(process.p, process.p1)

