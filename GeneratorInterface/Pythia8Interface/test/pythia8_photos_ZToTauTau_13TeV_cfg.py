import FWCore.ParameterSet.Config as cms

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.register('outFilename', 'particleLevel.root',  VarParsing.multiplicity.singleton, VarParsing.varType.string, "Output file name")
#options.register('photos', 'off', VarParsing.multiplicity.singleton, VarParsing.varType.string, "ME corrections")
options.register('lepton', 13, VarParsing.multiplicity.singleton, VarParsing.varType.int, "Lepton ID for Z decays")
options.register('cutoff', 0.00011, VarParsing.multiplicity.singleton, VarParsing.varType.float, "IR cutoff")
options.register('taufilter', 'off', VarParsing.multiplicity.singleton, VarParsing.varType.string, "Filter tau -> leptons")
options.parseArguments()
print(options)

process = cms.Process("PROD")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = int(options.maxEvents/100)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
    )
)

# set input to process
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source("EmptySource")

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.),
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'WeakSingleBoson:ffbar2gmZ = on',
            'PhaseSpace:mHatMin = 50.',
            '23:onMode = off',
            '23:onIfAny = 15',
            'TimeShower:mMaxGamma = 4.0',
            'TauDecays:externalMode = 0',
            'ParticleDecays:allowPhotonRadiation = on', # allow photons from hadron decays
            'TimeShower:QEDshowerByL = off', # no photons from leptons
            'TimeShower:QEDshowerByOther = off', # no photons from W bosons
            ),
		parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters')
    ),
      	ExternalDecays = cms.PSet(
        Photospp = cms.untracked.PSet(
            parameterSets = cms.vstring("setExponentiation", "setInfraredCutOff", "setCorrectionWtForW", "setMeCorrectionWtForW",
                                        "setMeCorrectionWtForZ", "setMomentumConservationThreshold", "setPairEmission", "setPhotonEmission",
                                        "setStopAtCriticalError", "suppressAll", "forceBremForDecay"),
            setExponentiation = cms.bool(True),
            setCorrectionWtForW = cms.bool(False),
            setMeCorrectionWtForW = cms.bool(False),
            setMeCorrectionWtForZ = cms.bool(False),
            setInfraredCutOff = cms.double(0.0000001),
            setMomentumConservationThreshold = cms.double(0.1),
            setPairEmission = cms.bool(False), # retain pair emission in MiNNLO x NLOEW / this
            setPhotonEmission = cms.bool(True),
            setStopAtCriticalError = cms.bool(False),
            # Use Photos only for W/Z and tau decays
            suppressAll = cms.bool(True),
            forceBremForDecay = cms.PSet(
                parameterSets = cms.vstring("Z", "Wp", "Wm", "tau", "atau"),
                Z = cms.vint32(0, 23),
                Wp = cms.vint32(0, 24),
                Wm = cms.vint32(0, -24),
                tau = cms.vint32(0, 15),
                atau = cms.vint32(0, -15)
            ),
	),
	parameterSets = cms.vstring("Photospp")
    )
)

## configure process options
process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True),
    wantSummary      = cms.untracked.bool(True)
)

process.genParticles = cms.EDProducer("GenParticleProducer",
    saveBarCodes = cms.untracked.bool(True),
    src = cms.InputTag("generator:unsmeared"),
    abortOnUnknownPDGCode = cms.untracked.bool(False)
)
process.printTree1 = cms.EDAnalyzer("ParticleListDrawer",
    src = cms.InputTag("genParticles"),
    maxEventsToPrint  = cms.untracked.int32(10)
)

process.path = cms.Path(process.generator)

