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
             "file:/data4/Wmunu_Summer09-MC_31X_V3_AODSIM-v1/0009/F82D4260-507F-DE11-B5D6-00093D128828.root")

)

process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")

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

# This Example uses only "corMetGlobalMuons". Modify to run over pf & tc Met (as "selectPfMetWMuNus")...
process.path = cms.Path(process.selectPfMetWMuNus)

# Maybe you want to comment the following sentence ;-)... 
process.end = cms.EndPath(process.wmnOutput)



