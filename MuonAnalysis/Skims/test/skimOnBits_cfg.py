import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIMONBITS")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'rfio:/castor/cern.ch/cms/store/caf/user/bellan/Run124120/MuonSkim_1.root'
    )
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.filter =cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","MUONSKIM"),
     HLTPaths = cms.vstring('muonTracksSkim'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
)

process.filterpath = cms.Path(process.filter)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('muonTrackSkim.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("filterpath")
    )
)

process.e = cms.EndPath(process.out)

