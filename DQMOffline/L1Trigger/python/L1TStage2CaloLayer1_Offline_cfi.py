#Online CaloL1 configuration, adjusted to now use offline module
#Andrew Loeliger <andrew.loeliger@cern.ch>
import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2CaloLayer1Offline = DQMEDAnalyzer('L1TStage2CaloLayer1Offline',
    ecalTPSourceRecd = cms.InputTag("caloLayer1Digis"),
    hcalTPSourceRecd = cms.InputTag("caloLayer1Digis"),
    ecalTPSourceSent = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    hcalTPSourceSent = cms.InputTag("hcalDigis"),
    fedRawDataLabel  = cms.InputTag("rawDataCollector"),
    histFolder = cms.string('L1T/L1TStage2CaloLayer1'),
    ignoreHFfb2 = cms.untracked.bool(False),
)
