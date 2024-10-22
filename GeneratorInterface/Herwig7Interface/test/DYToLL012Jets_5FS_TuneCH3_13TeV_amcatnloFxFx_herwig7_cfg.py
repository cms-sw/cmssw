# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Configuration/Generator/python/DYToLL012Jets_5FS_TuneCH3_13TeV_amcatnloFxFx_herwig7_cff.py --conditions auto:run2_mc -s LHE,GEN --datatier LHE,GEN -n 10 --eventcontent LHE,RAWSIM --no_exec
import FWCore.ParameterSet.Config as cms



process = cms.Process('GEN')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('Configuration/Generator/python/DYToLL012Jets_5FS_TuneCH3_13TeV_amcatnloFxFx_herwig7_cff.py nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.LHEoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('LHE'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('DYToLL012Jets_5FS_TuneCH3_13TeV_amcatnloFxFx_herwig7_cff_py_LHE_GEN.root'),
    outputCommands = process.LHEEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(1),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(20971520),
    fileName = cms.untracked.string('DYToLL012Jets_5FS_TuneCH3_13TeV_amcatnloFxFx_herwig7_cff_py_LHE_GEN_inRAWSIM.root'),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.generator = cms.EDFilter("Herwig7GeneratorFilter",
    configFiles = cms.vstring(),
    crossSection = cms.untracked.double(-1),
    dataLocation = cms.string('${HERWIGPATH:-6}'),
    eventHandlers = cms.string('/Herwig/EventHandlers'),
    filterEfficiency = cms.untracked.double(1.0),
    generatorModule = cms.string('/Herwig/Generators/EventGenerator'),
    herwig7CH3AlphaS = cms.vstring(
        'cd /Herwig/Shower', 
        'set AlphaQCD:AlphaIn 0.118', 
        'cd /'
    ),
    herwig7CH3MPISettings = cms.vstring(
        'set /Herwig/Hadronization/ColourReconnector:ReconnectionProbability 0.4712', 
        'set /Herwig/UnderlyingEvent/MPIHandler:pTmin0 3.04', 
        'set /Herwig/UnderlyingEvent/MPIHandler:InvRadius 1.284', 
        'set /Herwig/UnderlyingEvent/MPIHandler:Power 0.1362'
    ),
    herwig7CH3PDF = cms.vstring(
        'cd /Herwig/Partons', 
        'create ThePEG::LHAPDF PDFSet_nnlo ThePEGLHAPDF.so', 
        'set PDFSet_nnlo:PDFName NNPDF31_nnlo_as_0118.LHgrid', 
        'set PDFSet_nnlo:RemnantHandler HadronRemnants', 
        'set /Herwig/Particles/p+:PDF PDFSet_nnlo', 
        'set /Herwig/Particles/pbar-:PDF PDFSet_nnlo', 
        'set /Herwig/Partons/PPExtractor:FirstPDF  PDFSet_nnlo', 
        'set /Herwig/Partons/PPExtractor:SecondPDF PDFSet_nnlo', 
        'set /Herwig/Shower/ShowerHandler:PDFA PDFSet_nnlo', 
        'set /Herwig/Shower/ShowerHandler:PDFB PDFSet_nnlo', 
        'create ThePEG::LHAPDF PDFSet_lo ThePEGLHAPDF.so', 
        'set PDFSet_lo:PDFName NNPDF31_lo_as_0130.LHgrid', 
        'set PDFSet_lo:RemnantHandler HadronRemnants', 
        'set /Herwig/Shower/ShowerHandler:PDFARemnant PDFSet_lo', 
        'set /Herwig/Shower/ShowerHandler:PDFBRemnant PDFSet_lo', 
        'set /Herwig/Partons/MPIExtractor:FirstPDF PDFSet_lo', 
        'set /Herwig/Partons/MPIExtractor:SecondPDF PDFSet_lo', 
        'cd /'
    ),
    herwig7StableParticlesForDetector = cms.vstring(
        'set /Herwig/Decays/DecayHandler:MaxLifeTime 10*mm', 
        'set /Herwig/Decays/DecayHandler:LifeTimeOption Average'
    ),
    hw_mg_merging_settings = cms.vstring(
        'cd /Herwig/EventHandlers', 
        'library HwFxFx.so', 
        'create Herwig::FxFxEventHandler LesHouchesHandler', 
        'set LesHouchesHandler:PartonExtractor /Herwig/Partons/PPExtractor', 
        'set LesHouchesHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler', 
        'set LesHouchesHandler:DecayHandler /Herwig/Decays/DecayHandler', 
        'set LesHouchesHandler:WeightOption VarNegWeight', 
        'set /Herwig/Generators/EventGenerator:EventHandler  /Herwig/EventHandlers/LesHouchesHandler', 
        'create ThePEG::Cuts /Herwig/Cuts/NoCuts', 
        'cd /Herwig/EventHandlers', 
        'create Herwig::FxFxFileReader FxFxLHReader', 
        'insert LesHouchesHandler:FxFxReaders[0] FxFxLHReader', 
        'cd /Herwig/Shower', 
        'library HwFxFxHandler.so', 
        'create Herwig::FxFxHandler FxFxHandler', 
        'set /Herwig/Shower/FxFxHandler:SplittingGenerator /Herwig/Shower/SplittingGenerator', 
        'set /Herwig/Shower/FxFxHandler:KinematicsReconstructor /Herwig/Shower/KinematicsReconstructor', 
        'set /Herwig/Shower/FxFxHandler:PartnerFinder /Herwig/Shower/PartnerFinder', 
        'set /Herwig/EventHandlers/LesHouchesHandler:CascadeHandler /Herwig/Shower/FxFxHandler', 
        'set /Herwig/Partons/PDFSet_nnlo:PDFName NNPDF31_nnlo_as_0118', 
        'set /Herwig/Partons/RemnantDecayer:AllowTop Yes', 
        'set /Herwig/Partons/PDFSet_nnlo:RemnantHandler /Herwig/Partons/HadronRemnants', 
        'set /Herwig/Particles/p+:PDF /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Particles/pbar-:PDF /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Partons/PPExtractor:FirstPDF  /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Partons/PPExtractor:SecondPDF /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Shower/ShowerHandler:PDFA /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Shower/ShowerHandler:PDFB /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/EventHandlers/FxFxLHReader:FileName cmsgrid_final.lhe', 
        'set /Herwig/EventHandlers/FxFxLHReader:WeightWarnings false', 
        'set /Herwig/EventHandlers/FxFxLHReader:AllowedToReOpen No', 
        'set /Herwig/EventHandlers/FxFxLHReader:InitPDFs 0', 
        'set /Herwig/EventHandlers/FxFxLHReader:Cuts /Herwig/Cuts/NoCuts', 
        'set /Herwig/EventHandlers/FxFxLHReader:MomentumTreatment RescaleEnergy', 
        'set /Herwig/EventHandlers/FxFxLHReader:PDFA /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/EventHandlers/FxFxLHReader:PDFB /Herwig/Partons/PDFSet_nnlo', 
        'set /Herwig/Shower/ShowerHandler:MaxPtIsMuF Yes', 
        'set /Herwig/Shower/ShowerHandler:RestrictPhasespace Yes', 
        'set /Herwig/Shower/PartnerFinder:PartnerMethod Random', 
        'set /Herwig/Shower/PartnerFinder:ScaleChoice Partner', 
        'set /Herwig/Shower/KinematicsReconstructor:InitialInitialBoostOption LongTransBoost', 
        'set /Herwig/Shower/KinematicsReconstructor:ReconstructionOption General', 
        'set /Herwig/Shower/KinematicsReconstructor:InitialStateReconOption Rapidity', 
        'set /Herwig/Shower/ShowerHandler:SpinCorrelations Yes', 
        'cd /Herwig/Shower', 
        'set /Herwig/Shower/FxFxHandler:MPIHandler  /Herwig/UnderlyingEvent/MPIHandler', 
        'set /Herwig/Shower/FxFxHandler:RemDecayer  /Herwig/Partons/RemnantDecayer', 
        'set /Herwig/Shower/FxFxHandler:ShowerAlpha  AlphaQCD', 
        'set FxFxHandler:HeavyQVeto Yes', 
        'set FxFxHandler:HardProcessDetection Automatic', 
        'set FxFxHandler:drjmin 0', 
        'cd /Herwig/Shower', 
        'set FxFxHandler:VetoIsTurnedOff VetoingIsOn', 
        'set FxFxHandler:ETClus 20*GeV', 
        'set FxFxHandler:RClus 1.0', 
        'set FxFxHandler:EtaClusMax 10', 
        'set FxFxHandler:RClusFactor 1.5'
    ),
    hw_user_settings = cms.vstring(
        'set FxFxHandler:MergeMode FxFx', 
        'set FxFxHandler:njetsmax 2'
    ),
    parameterSets = cms.vstring(
        'herwig7CH3PDF', 
        'herwig7CH3AlphaS', 
        'herwig7CH3MPISettings', 
        'herwig7StableParticlesForDetector', 
        'hw_mg_merging_settings', 
        'hw_user_settings'
    ),
    repository = cms.string('${HERWIGPATH}/HerwigDefaults.rpo'),
    run = cms.string('InterfaceMatchboxTest'),
    runModeList = cms.untracked.string('read,run'),
    seed = cms.untracked.int32(12345)
)


process.externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/2017/13TeV/madgraph/V5_2.6.1/DYellell012j_5f_NLO_FXFX/dyellell012j_5f_NLO_FXFX_slc7_amd64_gcc700_CMSSW_10_6_4_tarball.tar.xz'),
    nEvents = cms.untracked.uint32(10),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)


process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.lhe_step = cms.Path(process.externalLHEProducer)
process.generation_step = cms.Path(process.pgen)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.LHEoutput_step = cms.EndPath(process.LHEoutput)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.lhe_step,process.generation_step,process.genfiltersummary_step,process.endjob_step,process.LHEoutput_step,process.RAWSIMoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)
# filter all path with the production filter sequence
for path in process.paths:
	if path in ['lhe_step']: continue
	getattr(process,path).insert(0, process.ProductionFilterSequence)



# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
