import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load('Configuration.StandardSequences.Services_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('SimGeneral.MixingModule.HiMixGEN_cff')
process.mix.digitizers = cms.PSet()
process.mix.LabelPlayback = 'mix'
process.mix.playback = True


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:step3_RAW2DIGI_L1Reco_RECO_PU.root'
    )
)

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("histograms_workflowD.root"))

process.load("GeneratorInterface.HiGenCommon.HeavyIon_cff")
process.demo = cms.EDAnalyzer('HiMixValidation',
                              jetSrc = cms.untracked.InputTag('akPu3CaloJets')
)

process.p = cms.Path(process.mix*process.heavyIon*process.demo)

for a in process.aliases: delattr(process, a)
