import FWCore.ParameterSet.Config as cms

l1demonstage1 = cms.EDAnalyzer("L1TDEMON",
    HistFolder = cms.untracked.string('L1TEMU'),
    HistFile = cms.untracked.string('l1demon.root'),
    disableROOToutput = cms.untracked.bool(True),
    DataEmulCompareSource = cms.InputTag("l1compareforstage1"),
    VerboseFlag = cms.untracked.int32(0),
    RunInFilterFarm = cms.untracked.bool(False),
    COMPARE_COLLS = cms.untracked.vuint32(
       0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  0
    # ETP,HTP,RCT,GCT,DTP,DTF,CTP,CTF,RPC,LTC,GMT,GT
    )
)


  
