import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("wmnsel")
process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(200)
      #input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
      #fileNames = cms.untracked.vstring("file:/data4/RelValWM_CMSSW_3_1_0-STARTUP31X_V1-v1_GEN-SIM-RECO/40BFAA1A-5466-DE11-B792-001D09F29533.root")
      #fileNames = cms.untracked.vstring("file:/data4/Wmunu-Summer09-MC_31X_V2_preproduction_311-v1/0011/F4C91F77-766D-DE11-981F-00163E1124E7.root")
       fileNames = cms.untracked.vstring("file:/data4/Wmunu_Summer09-MC_31X_V3_AODSIM-v1/0009/F82D4260-507F-DE11-B5D6-00093D128828.root")

)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('wmnSelFilter'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
            threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)


process.load("ElectroWeakAnalysis.WMuNu.wmunusValidation_cfi") #load validation sequence (for WMunu & ZMuMu)
process.wmnVal_pfMet.JetTag = cms.untracked.InputTag("antikt5PFJets")
#process.wmnVal_corMet.JetTag = cms.untracked.InputTag("antikt5CaloJets") 
#process.wmnVal_tcMet.JetTag = cms.untracked.InputTag("antikt5CaloJets") 


# Output
#process.load("Configuration.EventContent.EventContent_cff")
#process.wmnOutput = cms.OutputModule("PoolOutputModule",
#      process.AODSIMEventContent,
#      SelectEvents = cms.untracked.PSet(
#            SelectEvents = cms.vstring('wmnsel')
#      ),
#      fileName = cms.untracked.string('root_files/wmnsel.root')
#)

# Output histograms
process.TFileService = cms.Service("TFileService", fileName = cms.string('WMuNu_histograms.root') )

# Steering the process
process.wmnsel = cms.Path(process.wmnVal_pfMet)
#process.end = cms.EndPath(process.wmnOutput)
