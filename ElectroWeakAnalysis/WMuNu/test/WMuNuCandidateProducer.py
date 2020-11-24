import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("USER")
process.maxEvents = cms.untracked.PSet(
      #input = cms.untracked.int32(-1)
      input = cms.untracked.int32(20)
)

process.load("ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi")


process.source = cms.Source("PoolSource",
      debugVerbosity = cms.untracked.uint32(0),
      debugFlag = cms.untracked.bool(False),
      fileNames = cms.untracked.vstring("file:/data4/Wmunu_Summer09-MC_31X_V3_AODSIM-v1/0009/F82D4260-507F-DE11-B5D6-00093D128828.root")

)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring('corMetWMuNus')
)

# Output
process.load("Configuration.EventContent.EventContent_cff")

process.myEventContent = cms.PSet(
	outputCommands = cms.untracked.vstring(
		'keep *'
	)
)

process.wmnOutput = cms.OutputModule("PoolOutputModule",
      process.AODSIMEventContent,
      #process.myEventContent,
      SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring('path')
      ),
      fileName = cms.untracked.string('AOD_with_WCandidates.root')
)


# Steering the process
process.path = cms.Path(process.corMetWMuNus)
#process.path = cms.Path(process.pfMetWMuNus)
#process.path = cms.Path(process.tcMetWMuNus)
#process.path = cms.Path(process.allWMuNus)

process.end = cms.EndPath(process.wmnOutput)



