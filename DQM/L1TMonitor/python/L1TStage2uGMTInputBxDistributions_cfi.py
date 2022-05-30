import FWCore.ParameterSet.Config as cms

# the uGMT DQM module
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2uGMTInputBxDistributions = DQMEDAnalyzer(
    "L1TStage2uGMTInputBxDistributions",
    bmtfProducer = cms.InputTag("gmtStage2Digis", "BMTF"),
    omtfProducer = cms.InputTag("gmtStage2Digis", "OMTF"),
    emtfProducer = cms.InputTag("gmtStage2Digis", "EMTF"),
    emtfShowerProducer = cms.InputTag("gmtStage2Digis", "EMTF"),
    muonProducer = cms.InputTag("gmtStage2Digis", "Muon"),
    muonShowerProducer = cms.InputTag("gmtStage2Digis", "MuonShower"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT"),
    emulator = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False),
    hadronicShowers = cms.untracked.bool(False)
)

## Era: Run3_2021; Displaced muons from BMTF used in uGMT from Run-3
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(l1tStage2uGMTInputBxDistributions, hadronicShowers = cms.untracked.bool(True))
