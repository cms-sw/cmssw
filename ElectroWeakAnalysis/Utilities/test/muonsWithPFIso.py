import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("muonsWithPFIso")

process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
      #input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring("file:/ciet3b/data4/Spring10_10invpb_AODRED/Wmunu_1.root")
)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
      debugModules = cms.untracked.vstring('muonsWithPFIso'),
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(100) ),
            #threshold = cms.untracked.string('INFO')
            threshold = cms.untracked.string('DEBUG')
      ),
      destinations = cms.untracked.vstring('cout')
)

# Create a new reco::Muon collection with PFLow Iso information
process.muonsWithPFIso = cms.EDFilter("MuonWithPFIsoProducer",
        MuonTag = cms.untracked.InputTag("muons")
      , PfTag = cms.untracked.InputTag("particleFlow")
      , UsePfMuonsOnly = cms.untracked.bool(False)
      , TrackIsoVeto = cms.untracked.double(0.01)
      , GammaIsoVeto = cms.untracked.double(0.07)
      , NeutralHadronIsoVeto = cms.untracked.double(0.1)
)

# WMN fast selector (use W candidates in this example)
process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")
process.pfMetWMuNus.MuonTag = cms.untracked.InputTag("muonsWithPFIso")
process.selpfMet.MuonTag = cms.untracked.InputTag("muonsWithPFIso")
# Use the following line only for old Summer09 samples (new: "ak5", old "antikt5")
#process.selpfMet.JetTag = cms.untracked.InputTag("antikt5PFJets") 

# Output
process.load("Configuration.EventContent.EventContent_cff")
process.AODSIMEventContent.outputCommands.append('keep *_muonsWithPFIso_*_*')
process.myEventContent = process.AODSIMEventContent
process.wmnOutput = cms.OutputModule("PoolOutputModule",
      process.myEventContent,
      SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring('muonsWithPFIsoSelection')
      ),
      fileName = cms.untracked.string('selectedEvents.root')
)

# Steering the process
process.muonsWithPFIsoSelection = cms.Path(
       process.muonsWithPFIso*
       process.selectPfMetWMuNus
)

process.end = cms.EndPath(process.wmnOutput)
