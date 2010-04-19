# WMuNuAnalysis configuration file
# cfg file to use if you want "everything" on one go

import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("wmnsel")
process.maxEvents = cms.untracked.PSet(
      #input = cms.untracked.int32(-1)
      input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
       fileNames = cms.untracked.vstring(
#      'file:/data1/degrutto/CMSSW_3_5_6/src/ElectroWeakAnalysis/Skimming/test/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133483_3.root'
   'file:/ciet3a/data4/EWK_SubSkim_Summer09_7TeV/WMuNu_7TeV_10invpb_1.root',

        )
)

process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")  # standard cuts defined in cff file
#process.selcorMet.plotHistograms = cms.untracked.bool(True) 
process.selpfMet.plotHistograms = cms.untracked.bool(True) # --> "true" for plotting of histos
#process.seltcMet.plotHistograms = cms.untracked.bool(True)

 
#process.load("ElectroWeakAnalysis.WMuNu.wmunusValidation_cfi") #load validation sequence (for WMunu & ZMuMu)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('selpfMet'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(100) ),
            threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

# Output histograms
process.TFileService = cms.Service("TFileService", fileName = cms.string('WMuNu_prueba.root') )

# Steering the process
#process.path0 = cms.Path(process.wmunuVal) # This creates extra validation folders, not strictly necesary for analysis
process.path1 = cms.Path(process.selectPfMetWMuNus) #--> This is the default now!
#process.path2 = cms.Path(process.selectTcMetWMuNus)
#process.path3 = cms.Path(process.selectCaloMetWMuNus)



