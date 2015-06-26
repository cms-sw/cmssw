import FWCore.ParameterSet.Config as cms

process = cms.Process('GEN')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

#process.MessageLogger.cerr.FwkReport.reportEvery = 10

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source('LHESource',
    fileNames = cms.untracked.vstring('file:w01j_5f_NLO.lhe')
)

process.load('Configuration.Generator.HerwigppDefaults_cfi')

process.generator = cms.EDFilter('ThePEGHadronizerFilter',
    process.herwigDefaultsBlock,
    configFiles = cms.vstring(),
    lheDefaults1 = cms.vstring(
         'cd /Herwig/Cuts',
         'create ThePEG::Cuts NoCuts',
         'cd /Herwig/EventHandlers',
         'create ThePEG::LesHouchesInterface LHEReader',
         'set LHEReader:Cuts /Herwig/Cuts/NoCuts',
         'create ThePEG::LesHouchesEventHandler LHEHandler',
#         'set LHEHandler:WeightOption VarWeight',
         'set LHEHandler:WeightOption 1',
         'set LHEHandler:PartonExtractor /Herwig/Partons/QCDExtractor',
         'set LHEHandler:CascadeHandler /Herwig/Shower/ShowerHandler',
         'set LHEHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler',
         'set LHEHandler:DecayHandler /Herwig/Decays/DecayHandler',
         'insert LHEHandler:LesHouchesReaders 0 LHEReader',
         'cd /Herwig/Generators',
         'set LHCGenerator:EventHandler /Herwig/EventHandlers/LHEHandler',
         'cd /Herwig/Shower',
         'set Evolver:HardVetoScaleSource Read',
         'set Evolver:MECorrMode No',
         'cd /',
    ),
    lheDefaultPDFs1 = cms.vstring(
#         'set /Herwig/EventHandlers/LHEReader:PDFA /Herwig/Partons/myPDFset',
#         'set /Herwig/EventHandlers/LHEReader:PDFB /Herwig/Partons/myPDFset',
    ),
    parameterSets = cms.vstring(
        'cmsDefaults',  # NOTE: pp@14TeV by default
        'lheDefaults1',
        #'cm14TeV',
    ),
)

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('file:HerwigppHadronizerOutput.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN')
    )
)

process.generation_step = cms.Path(process.pgen)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.output_step = cms.EndPath(process.RAWSIMoutput)

process.schedule = cms.Schedule(process.generation_step,process.endjob_step,process.output_step)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.generator * getattr(process,path)._seq
