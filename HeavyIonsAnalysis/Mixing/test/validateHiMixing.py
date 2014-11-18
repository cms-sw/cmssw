import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load('Configuration.StandardSequences.Services_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('HeavyIonsAnalysis.Mixing.HiMix_cff')
process.mix = process.mixVal.clone()

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:output_workflowD_step3_4.root'
    )
)

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("histograms_workflowD.root"))

process.load("GeneratorInterface.HiGenCommon.HeavyIon_cff")
process.demo = cms.EDAnalyzer('HiMixValidation')

process.p = cms.Path(process.mix*process.heavyIon*process.demo)

for a in process.aliases: delattr(process, a)
