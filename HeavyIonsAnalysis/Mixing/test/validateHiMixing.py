import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:output_workflowD_step1_0'
    )
)

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("histograms_workflowD.root"))

process.load("GeneratorInterface.HiGenCommon.HeavyIon_cff")
process.demo = cms.EDAnalyzer('HiMixValidation')

process.p = cms.Path(process.heavyIon*process.demo)
