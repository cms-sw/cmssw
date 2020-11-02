import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(3),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    comEnergy = cms.double(7000.),
    
    ExternalDecays = cms.PSet(
        EvtGen1 = cms.untracked.PSet(
             decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2010.DEC'),
             particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
             #user_decay_files = cms.vstring('DECAY_2010.DEC'),
             user_decay_file = cms.vstring('GeneratorInterface/ExternalDecays/data/Onia_mumu.dec'),
             list_forced_decays = cms.vstring('MyUpsilon(2S)'),
             operates_on_particles = cms.vint32(0) # 0 (zero) means default list (hardcoded), the list of PDG IDs can be put here
             ),
        parameterSets = cms.vstring('EvtGen1')
    ),
    
    PythiaParameters = cms.PSet(      # MultipleInteractions commands below don't work (obsolete?). Should official settings be used?

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
				     # without messing up too much the principle physics
				     'PartonLevel:MPI = off',            # D; mstp(81)=1
				     ### Py6 mstp(82)=4 (MI model) does NOT have anything similar in Py8, 
				     ### because it actually means a new model, and in Py8 old models went away 
				     ### - only new remain
				     #'MultipleInteractions:pT0Ref=1.8387',    # parp(82)=1.8387 - pt cutoff for MI
				     #'MultipleInteractions:ecmRef=1960.',     # parp(89)=1960.
				     #'MultipleInteractions:coreFraction=0.5', # D; parp(83)=0.5
				     #'MultipleInteractions:coreRadius=0.4',   # D; parp(84)=0.4
				     ### parp(85) & parp(86) are N/A in Py8
				     #'MultipleInteractions:ecmPow=0.16',      # parp(90)=0.16
				     'BeamRemnants:primordialKT=on',          # D; mstp(91)=1
				     'BeamRemnants:primordialKThard=2.1',     # parp(91)=2.1
				     'BeamRemnants:primordialKTremnant=15.',  # parp(93)=15.
				     'SpaceShower:pTmin=1.25',                # parp(62)=1.25
				     ### parp(64) is N/A in Py8
				     'SpaceShower:pTmaxFudge=2.'              # should this be instead of parp(67)=2.5 ??? 
				                                              # In Py6, D==parp(67)=4.
				                                              # and in Py8 for this D=1, max=2
	),
        py8ProcessSettings = cms.vstring( 'Bottomonium:all = on' # Quarkonia (bottonium/NRQCD) - msel=62
#        py8ProcessSettings = cms.vstring( 'Bottomonium:gg2QQbar[3S1(1)]g = on' # Quarkonia - msel=62, isub=???
	),
	parameterSets = cms.vstring('py8UESettings','py8ProcessSettings')
    )
)

# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []


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
    fileName = cms.untracked.string('Py8Onia_EvtGen.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)

