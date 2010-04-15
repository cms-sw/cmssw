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
        'file:/data2/Wmunu_Summer09-MC_31X_V3_7TeV_AODSIM-v1/002861FD-899B-DE11-915E-000AE488BE67.root'
        )
)

process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")  # standard cuts defined in cff file
#process.selcorMet.plotHistograms = cms.untracked.bool(True) 
process.selpfMet.plotHistograms = cms.untracked.bool(True) # --> "true" for plotting of histos
#process.seltcMet.plotHistograms = cms.untracked.bool(True)
 
#process.load("ElectroWeakAnalysis.WMuNu.wmunusValidation_cfi") #load validation sequence (for WMunu & ZMuMu)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('selectPfMetWMuNus'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(100) ),
            threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

# Output histograms
process.TFileService = cms.Service("TFileService", fileName = cms.string('WMuNu.root') )

# Steering the process
#process.path0 = cms.Path(process.wmunuval) # This creates extra validation folders, not strictly necesary for analysis
process.path1 = cms.Path(process.selectPfMetWMuNus) #--> This is the default now!
#process.path2 = cms.Path(process.selectTcMetWMuNus)
#process.path3 = cms.Path(process.selectCaloMetWMuNus)

