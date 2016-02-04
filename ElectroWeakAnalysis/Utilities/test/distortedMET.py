import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("distortMET")

process.maxEvents = cms.untracked.PSet(
      #input = cms.untracked.int32(-1)
      input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring("file:/data4/Wmunu_Summer09-MC_31X_V3_AODSIM-v1/0009/F82D4260-507F-DE11-B5D6-00093D128828.root")
)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(100) ),
            threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

# Selector and parameters
process.distortedMET = cms.EDFilter("DistortedMETProducer",
      MetTag = cms.untracked.InputTag("met"),
      MetScaleShift = cms.untracked.double(0.2)
)

### NOTE: the following WMN selectors require the presence of
### the libraries and plugins fron the ElectroWeakAnalysis/WMuNu package
### So you need to process the ElectroWeakAnalysis/WMuNu package with
### some old CMSSW versions (at least <=3_1_2, <=3_3_0_pre4)
#

# WMN fast selector (use W candidates in this example)
process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")

# Output
process.load("Configuration.EventContent.EventContent_cff")
process.AODSIMEventContent.outputCommands.append('keep *_distortedMET_*_*')
process.myEventContent = process.AODSIMEventContent
process.wmnOutput = cms.OutputModule("PoolOutputModule",
      process.myEventContent,
      SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring('distortMET')
      ),
      fileName = cms.untracked.string('selectedEvents.root')
)

# Steering the process
process.distortMET = cms.Path(
       process.distortedMET
      *process.selectCaloMetWMuNus
)
process.end = cms.EndPath(process.wmnOutput)
