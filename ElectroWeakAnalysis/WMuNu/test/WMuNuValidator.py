import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("wmnsel")
process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
      #input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
       fileNames = cms.untracked.vstring('/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/167/786/401AE9B7-F8A1-E011-93CB-003048F1C832.root')
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
