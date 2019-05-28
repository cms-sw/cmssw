import FWCore.ParameterSet.Config as cms

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.register('outFilename', 'particleLevel.root',  VarParsing.multiplicity.singleton, VarParsing.varType.string, "Output file name")
options.register('photos', 'off', VarParsing.multiplicity.singleton, VarParsing.varType.string, "ME corrections")
options.register('lepton', 13, VarParsing.multiplicity.singleton, VarParsing.varType.int, "Lepton ID for Z decays")
options.setDefault('maxEvents', 10000)
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

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(7000.),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring('WeakSingleBoson:ffbar2gmZ = on',
                                        'PhaseSpace:mHatMin = 50.',
                                        '23:onMode = off', 
                                        '23:onIfAny = %i' % options.lepton,
                                        #'PartonLevel:MPI = off',
                                        #'PartonLevel:ISR = off',
                                        #'PartonLevel:FSR = off',
                                        #'HadronLevel:all = off',
                                        ),
        parameterSets = cms.vstring('processParameters')
    )
)

if options.photos == 'exp':
    process.generator.ExternalDecays = cms.PSet(
        Photospp = cms.untracked.PSet(
            parameterSets = cms.vstring("setExponentiation", "setInfraredCutOff", "setMeCorrectionWtForW", "setMeCorrectionWtForZ", "setMomentumConservationThreshold", "setPairEmission", "setPhotonEmission", "setStopAtCriticalError"),
            setExponentiation = cms.bool(True),
            setMeCorrectionWtForW = cms.bool(True),
            setMeCorrectionWtForZ = cms.bool(True),
            setInfraredCutOff = cms.double(0.00011),
            setMomentumConservationThreshold = cms.double(0.1),
            setPairEmission = cms.bool(True),
            setPhotonEmission = cms.bool(True),
            setStopAtCriticalError = cms.bool(False),
        ),
        parameterSets = cms.vstring("Photospp")
    )
    process.generator.PythiaParameters.processParameters += cms.vstring(
        'ParticleDecays:allowPhotonRadiation = off',
        'TimeShower:QEDshowerByL = off',
    )

if options.photos == 'single':
    process.generator.ExternalDecays = cms.PSet(
        Photospp = cms.untracked.PSet(
            parameterSets = cms.vstring("setExponentiation", "setInfraredCutOff", "setMeCorrectionWtForW", "setMeCorrectionWtForZ", "setMomentumConservationThreshold", "setPairEmission", "setPhotonEmission", "setStopAtCriticalError"),
            setExponentiation = cms.bool(False),
            setInfraredCutOff = cms.double(0.001),
            setMeCorrectionWtForW = cms.bool(True),
            setMeCorrectionWtForZ = cms.bool(True),
            setMomentumConservationThreshold = cms.double(0.1),
            setPairEmission = cms.bool(True),
            setPhotonEmission = cms.bool(True),
            setStopAtCriticalError = cms.bool(False),
        ),
        parameterSets = cms.vstring("Photospp")
    )
    process.generator.PythiaParameters.processParameters += cms.vstring(
        'ParticleDecays:allowPhotonRadiation = off',
        'TimeShower:QEDshowerByL = off'
    )

if options.photos == 'double':
    process.generator.ExternalDecays = cms.PSet(
        Photospp = cms.untracked.PSet(
            parameterSets = cms.vstring("setExponentiation", "setDoubleBrem", "setInfraredCutOff", "setMeCorrectionWtForW", "setMeCorrectionWtForZ", "setMomentumConservationThreshold", "setPairEmission", "setPhotonEmission", "setStopAtCriticalError"),
            setExponentiation = cms.bool(False),
            setDoubleBrem = cms.bool(True),
            setInfraredCutOff = cms.double(0.001),
            setMeCorrectionWtForW = cms.bool(True),
            setMeCorrectionWtForZ = cms.bool(True),
            setMomentumConservationThreshold = cms.double(0.1),
            setPairEmission = cms.bool(True),
            setPhotonEmission = cms.bool(True),
            setStopAtCriticalError = cms.bool(False),
        ),
        parameterSets = cms.vstring("Photospp")
    )
    process.generator.PythiaParameters.processParameters += cms.vstring(
        'ParticleDecays:allowPhotonRadiation = off',
        'TimeShower:QEDshowerByL = off'
    )

if options.photos == 'quatro':
    process.generator.ExternalDecays = cms.PSet(
        Photospp = cms.untracked.PSet(
            parameterSets = cms.vstring("setExponentiation", "setQuatroBrem", "setInfraredCutOff", "setMeCorrectionWtForW", "setMeCorrectionWtForZ", "setMomentumConservationThreshold", "setPairEmission", "setPhotonEmission", "setStopAtCriticalError"),
            setExponentiation = cms.bool(False),
            setQuatroBrem = cms.bool(True),
            setInfraredCutOff = cms.double(0.001),
            setMeCorrectionWtForW = cms.bool(True),
            setMeCorrectionWtForZ = cms.bool(True),
            setMomentumConservationThreshold = cms.double(0.1),
            setPairEmission = cms.bool(True),
            setPhotonEmission = cms.bool(True),
            setStopAtCriticalError = cms.bool(False),
        ),
        parameterSets = cms.vstring("Photospp")
    )
    process.generator.PythiaParameters.processParameters += cms.vstring(
        'ParticleDecays:allowPhotonRadiation = off',
        'TimeShower:QEDshowerByL = off'
    )

if options.photos == 'nofsr':
    process.generator.PythiaParameters.processParameters += cms.vstring(
        'TimeShower:QEDshowerByL = off'
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

process.load("GeneratorInterface.RivetInterface.particleLevel_cfi")
process.particleLevel.src = cms.InputTag("generator:unsmeared")
process.particleLevel.lepConeSize = 0.1

process.load("GeneratorInterface.RivetInterface.rivetAnalyzer_cfi")
process.rivetAnalyzer.HepMCCollection = cms.InputTag("generator:unsmeared")
process.rivetAnalyzer.AnalysisNames = cms.vstring('CMS_2015_I1346843', 'MC_ZINC_MU', 'MC_ZINC_MU_BARE', 'MC_ZINC_EL', 'MC_ZINC_EL_BARE', 'MC_PHOTONS', 'MC_MUONS', 'MC_ELECTRONS')
process.rivetAnalyzer.OutputFile = cms.string('run.yoda')

process.path = cms.Path(process.generator*process.rivetAnalyzer)#*process.particleLevel*process.genParticles*process.printTree1)

