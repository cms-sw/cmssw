import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("USER")
process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
)

process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")
process.load("ElectroWeakAnalysis.WMuNu.wmunusValidation_cfi")

process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring(
        'file:/ciet3b/data3/GoldenWmunus/GoldenWmunus_ICHEP10.root'
      ),
      inputCommands = cms.untracked.vstring(
      'keep *', 'drop *_lumiProducer_*_*', 'drop *_MEtoEDMConverter_*_*', 'drop *_l1GtTriggerMenuLite_*_*' 
      )
)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('selcorMet'),
      cout = cms.untracked.PSet(
             default = cms.untracked.PSet( limit = cms.untracked.int32(20) ),
             threshold = cms.untracked.string('ERROR')
       #      threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

process.TFileService = cms.Service("TFileService", fileName = cms.string('WMuNu.root') )

process.selcorMet.plotHistograms = cms.untracked.bool(True)
process.selpfMet.plotHistograms = cms.untracked.bool(True)
process.seltcMet.plotHistograms = cms.untracked.bool(True)

# Steering the process
process.path1 = cms.Path(process.wmnVal_corMet)
process.path2 = cms.Path(process.wmnVal_pfMet)
process.path3 = cms.Path(process.wmnVal_tcMet)

process.path5 = cms.Path(process.selectCaloMetWMuNus)
process.path6 = cms.Path(process.selectPfMetWMuNus)
process.path7 = cms.Path(process.selectTcMetWMuNus)

