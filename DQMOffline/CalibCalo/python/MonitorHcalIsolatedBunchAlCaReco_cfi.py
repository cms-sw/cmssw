# The following comments couldn't be translated into the new config version:

import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
HcalIsolatedBunchMon = DQMEDAnalyzer('DQMHcalIsolatedBunchAlCaReco',
    # product to monitor
    hbheInput     = cms.InputTag("hbhereco"),
    hoInput       = cms.InputTag("horeco"),
    hfInput       = cms.InputTag("hfreco"),
    TriggerResult = cms.InputTag("TriggerResults","","HLT"),
    TriggerName   = cms.string("HLT_HcalIsolatedBunch"),
    PlotAll       = cms.untracked.bool(False),
    FolderName    = cms.untracked.string('AlCaReco/HcalIsolatedBunch')
)



