import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("USER")
process.maxEvents = cms.untracked.PSet(
      #input = cms.untracked.int32(-1)
      input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
      debugVerbosity = cms.untracked.uint32(0),
      debugFlag = cms.untracked.bool(False),
      fileNames = cms.untracked.vstring(
      #           "file:/data4/InclusiveMu15_Summer09-MC_31X_V3_AODSIM-v1/0024/C2F408ED-E181-DE11-8949-0030483344E2.root")
      #       "file:/data4/Wmunu_Summer09-MC_31X_V3_AODSIM-v1/0009/F82D4260-507F-DE11-B5D6-00093D128828.root")
              '/store/user/cepeda/mytestSkim_Wmunu_10pb/EWK_WMuNu_SubSkim_31Xv3_1.root' 

)

process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('corMetWMuNus','selcorMet'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
            threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

#process.TFileService = cms.Service("TFileService", fileName = cms.string('WMuNu.root') )

process.myEventContent = cms.PSet(
      outputCommands = cms.untracked.vstring(
            'keep *'
      )
)

process.wmnOutput = cms.OutputModule("PoolOutputModule",
      #process.AODSIMEventContent,
      process.myEventContent,
      SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring('path')
      ),
      fileName = cms.untracked.string('AOD_with_WCandidates.root')
)

# This Example uses only "corMetGlobalMuons". Modify to run over pf & tc Met
process.path = cms.Path(process.selectCaloMetWMuNus)

process.end = cms.EndPath(process.wmnOutput)



