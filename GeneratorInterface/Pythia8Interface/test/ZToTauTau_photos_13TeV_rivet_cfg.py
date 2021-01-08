import FWCore.ParameterSet.Config as cms

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.register('outFilename', 'particleLevel.root',  VarParsing.multiplicity.singleton, VarParsing.varType.string, "Output file name")
options.register('photos', 'off', VarParsing.multiplicity.singleton, VarParsing.varType.string, "ME corrections")
options.register('lepton', 13, VarParsing.multiplicity.singleton, VarParsing.varType.int, "Lepton ID for Z decays")
options.register('cutoff', 0.00011, VarParsing.multiplicity.singleton, VarParsing.varType.float, "IR cutoff")
options.register('taufilter', 'off', VarParsing.multiplicity.singleton, VarParsing.varType.string, "Filter tau -> leptons")
options.setDefault('maxEvents', 100)
options.parseArguments()
print(options)

process = cms.Process("PROD")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = int(options.maxEvents/100)

from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randSvc.populate()

# set input to process
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

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
            'ParticleDecays:allowPhotonRadiation = on',
            'TimeShower:QEDshowerByL = off',
            ),
		parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters')
    ),
	ExternalDecays = cms.PSet(
        Photospp = cms.untracked.PSet(
            parameterSets = cms.vstring("setExponentiation", "setInfraredCutOff", "setMeCorrectionWtForW", "setMeCorrectionWtForZ", "setMomentumConservationThreshold", "setPairEmission", "setPhotonEmission", "setStopAtCriticalError", "suppressAll", "forceBremForDecay"),
            setExponentiation = cms.bool(True),
            setMeCorrectionWtForW = cms.bool(True),
            setMeCorrectionWtForZ = cms.bool(True),
            setInfraredCutOff = cms.double(0.00011),
            setMomentumConservationThreshold = cms.double(0.1),
            setPairEmission = cms.bool(True),
            setPhotonEmission = cms.bool(True),
            setStopAtCriticalError = cms.bool(False),
            # Use Photos only for W/Z decays
            suppressAll = cms.bool(True),
            forceBremForDecay = cms.PSet(
                parameterSets = cms.vstring("Z", "Wp", "Wm"),
                Z = cms.vint32(0, 23),
                Wp = cms.vint32(0, 24),
                Wm = cms.vint32(0, -24),
            ),
        ),
        parameterSets = cms.vstring("Photospp")
    )
)

if options.taufilter != 'off':
    process.generator.PythiaParameters.processParameters.append('BiasedTauDecayer:filter = on')
    if options.taufilter == 'el':
        process.generator.PythiaParameters.processParameters.append('BiasedTauDecayer:eDecays = on')
        process.generator.PythiaParameters.processParameters.append('BiasedTauDecayer:muDecays = off')
    if options.taufilter == 'mu':
        process.generator.PythiaParameters.processParameters.append('BiasedTauDecayer:eDecays = off')
        process.generator.PythiaParameters.processParameters.append('BiasedTauDecayer:muDecays = on')


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

process.load("GeneratorInterface.RivetInterface.rivetAnalyzer_cfi")
process.rivetAnalyzer.AnalysisNames = cms.vstring('PDG_TAUS', 'MC_ELECTRONS', 'MC_MUONS', 'MC_TAUS')
process.rivetAnalyzer.useLHEweights = False
# process.rivetAnalyzer.OutputFile = 'out.yoda'
process.rivetAnalyzer.OutputFile = 'z-taufilter-%s.yoda' % options.taufilter

# process.path = cms.Path(process.externalLHEProducer*process.generator*process.rivetAnalyzer)
process.path = cms.Path(process.generator*process.rivetAnalyzer)

