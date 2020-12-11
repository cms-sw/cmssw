#
# cfg file to run L1GmtTriggerSource
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("TEST")

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RECO/v1/000/066/740/20DC42CA-0A9E-DD11-ADA0-001617E30D12.root')
)

process.l1GmtTriggerSource = cms.EDAnalyzer("L1GmtTriggerSource",
    GMTInputTag = cms.InputTag("gtDigis")
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    )
)

# path to be run
process.p = cms.Path(process.l1GmtTriggerSource)

