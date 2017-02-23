import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
            'file:GluGluTo2Jets_M_200_3000_13TeV_exhume_cff_py_GEN_SIM_RECOBEFMIX_DIGI_RECO_NoPileUp_Run2_25ns_CMSSW_8_0_0_pre2.root'
    )
)

process.demo = cms.EDAnalyzer('Validation'
		, jets      = cms.InputTag('ak4PFJets')
        , ppsReco   = cms.InputTag('ppssim:PPSReco')
        , ppsSim   = cms.InputTag('ppssim:PPSSim')
        , ppsGen   = cms.InputTag('ppssim:PPSGen')
)


process.TFileService = cms.Service("TFileService",
                fileName = cms.string('histo_validation_FastSim_CTPPS_pps.root')
        )



process.p = cms.Path(process.demo)
