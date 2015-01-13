import FWCore.ParameterSet.Config as cms

process = cms.Process('GEN')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# print event number once every 1000 events
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source('LHESource',
    fileNames = cms.untracked.vstring('file:/data4/juwu/LHE/dy0j_5f_LO_MLM.lhe')
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'POSTLS172_V7::All', '')

process.load('Configuration.Generator.HerwigppDefaults_cfi')

process.generator = cms.EDFilter('ThePEGHadronizerFilter',
    process.herwigDefaultsBlock,
    configFiles = cms.vstring(),
    parameterSets = cms.vstring(
        'cmsDefaults',  # NOTE: pp@14TeV by default
        'lheDefaults',
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

for path in process.paths:
    getattr(process,path)._seq = process.generator * getattr(process,path)._seq
