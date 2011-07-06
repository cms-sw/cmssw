import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("wmnsel")
process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
      #input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
<<<<<<< WMuNuValidator.py
       fileNames = cms.untracked.vstring(
"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre1/RelValWM/GEN-SIM-RECO/MC_42_V7-v1/0048/6AA13071-0859-E011-A86A-0018F3D09688.root"
)
=======
       fileNames = cms.untracked.vstring('/store/data/Run2011A/SingleMu/AOD/PromptReco-v4/000/167/786/401AE9B7-F8A1-E011-93CB-003048F1C832.root')
>>>>>>> 1.8
)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('wmnsel'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
            #threshold = cms.untracked.string('INFO')
            threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)


process.load("ElectroWeakAnalysis.WMuNu.wmunusValidation_cfi") #load validation sequence (for WMunu & ZMuMu)
#process.load("UserCode.MCepeda.GoodData2010_cfi")
process.wmnVal_pfMet.MuonTrig = cms.untracked.vstring("HLT_Mu9","HLT_Mu11","HLT_Mu15_v")


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
