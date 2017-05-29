import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        #'file:myfile.root'
    )
)

process.demo = cms.EDAnalyzer('Validation'
		, jets      = cms.InputTag('ak4PFJets')
		, vertices      = cms.InputTag('offlinePrimaryVertices')
		, muons      = cms.InputTag('muons')
		, electrons      = cms.InputTag('gedGsfElectrons')
		, ppsGen   = cms.InputTag('ppssim:PPSGen')
		, ppsSim   = cms.InputTag('ppssim:PPSSim')
		, ppsReco   = cms.InputTag('ppssim:PPSReco')
)

process.TFileService = cms.Service("TFileService",
                fileName = cms.string('histo_ValidationCMSPPS.root')
        )

process.p = cms.Path(process.demo)
