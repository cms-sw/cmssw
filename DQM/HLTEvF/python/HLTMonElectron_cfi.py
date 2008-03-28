import FWCore.ParameterSet.Config as cms

hltMonElectron = cms.EDFilter("HLTMonElectron",
    HLTCollectionLabels = cms.VInputTag(cms.InputTag("l1seedRelaxedSingle"), cms.InputTag("hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional"), cms.InputTag("hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter")),
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    theHLTOutputTypes = cms.vint32(82, 100, 100),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)


