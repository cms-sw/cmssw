import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hltTauOfflineCertification = DQMEDHarvester("HLTTauCertifier",
                                   targetDir = cms.string("HLT/EventInfo/reportSummaryContents"),
                                   targetME  = cms.string("HLT_Tau"),
                                   inputMEs = cms.vstring(
                                      "HLT/TauOffline/Inclusive/DoubleTau/TriggerBits",
                                   ),
                                   setBadRunOnWarnings = cms.bool(False),
                                   setBadRunOnErrors   = cms.bool(True)
)



