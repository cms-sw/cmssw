import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("distortMET")

process.maxEvents = cms.untracked.PSet(
      #input = cms.untracked.int32(-1)
      input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
      debugVerbosity = cms.untracked.uint32(0),
      debugFlag = cms.untracked.bool(False),
      fileNames = cms.untracked.vstring("file:/data4/Wmunu-Summer09-MC_31X_V2_preproduction_311-v1/0011/F4C91F77-766D-DE11-981F-00163E1124E7.root")
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
      MetScaleShift = cms.untracked.double(0.1)
)

### NOTE: the following WMN selectors require the presence of
### the libraries and plugins fron the ElectroWeakAnalysis/WMuNu package
### So you need to process the ElectroWeakAnalysis/WMuNu package with
### some old CMSSW versions (at least <=3_1_2, <=3_3_0_pre4)
#

# WMN fast selector (use W candidates in this example)
process.corMetWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("corMetGlobalMuons"),
      METIncludesMuons = cms.untracked.bool(True),
      JetTag = cms.untracked.InputTag("sisCone5CaloJets"),
)

process.wmnSelFilter = cms.EDFilter("WMuNuSelector",
      # Fill Basic Histograms? ->
      plotHistograms = cms.untracked.bool(False),

      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("corMetGlobalMuons"),
      METIncludesMuons = cms.untracked.bool(True),
      JetTag = cms.untracked.InputTag("sisCone5CaloJets"),
      WMuNuCollectionTag = cms.untracked.InputTag("corMetWMuNus:WMuNuCandidates")
)

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
      *process.corMetWMuNus
      *process.wmnSelFilter
)
process.end = cms.EndPath(process.wmnOutput)
