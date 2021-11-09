# The configuration for comparison of Pythia8 tau decays with Taula tau decays (Py8_Z2tautau_tauola_cfg.py)
# hadronization, MPI settings, PDF settings to be eventually updated in both
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
        py8UESettings = cms.vstring( 'ParticleDecays:limitTau0 = on',  # - decay unstable particles
				     'ParticleDecays:tau0Max   = 10.', # - for which ctau < 10 mm
				     'HadronLevel:Hadronize = on',
				     'PartonLevel:MPI = on'
	),
        py8ProcessSettings = cms.vstring( 'WeakSingleBoson:ffbar2gmZ = on'
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
