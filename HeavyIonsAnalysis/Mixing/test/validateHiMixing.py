import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:myfile.root'
    )
)

process.load("GeneratorInterface.HiGenCommon.HeavyIon_cff")
process.demo = cms.EDAnalyzer('HiMixValidation')

process.p = cms.Path(process.heavyIon*process.demo)
