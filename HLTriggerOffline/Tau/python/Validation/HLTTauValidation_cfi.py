import FWCore.ParameterSet.Config as cms

hltTauValidationProcess_IDEAL = "HLT"

hltTauValIdealMonitorMC = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    HLTProcessName = cms.untracked.string(hltTauValidationProcess_IDEAL),
    DQMBaseFolder = cms.untracked.string("HLT/TauRelVal/MC"),
    TriggerResultsSrc = cms.untracked.InputTag("TriggerResults", "", hltTauValidationProcess_IDEAL),
    TriggerEventSrc = cms.untracked.InputTag("hltTriggerSummaryAOD", "", hltTauValidationProcess_IDEAL),
    L1Plotter = cms.untracked.PSet(
        DQMFolder             = cms.untracked.string('L1'),
        L1Taus                = cms.untracked.InputTag("caloStage2Digis", "Tau"),
        L1ETM                 = cms.untracked.InputTag("caloStage2Digis","EtSum"),
        L1ETMMin              = cms.untracked.double(50),
    ),
    Paths = cms.untracked.string("PFTau"),
    PathSummaryPlotter = cms.untracked.PSet(
        DQMFolder             = cms.untracked.string('Summary'),
    ),
    Matching = cms.PSet(
        doMatching            = cms.untracked.bool(True),
        matchFilters          = cms.untracked.VPSet(
                                    cms.untracked.PSet(
                                        FilterName        = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
                                        matchObjectID     = cms.untracked.int32(15),
                                    ),
                                    cms.untracked.PSet(
                                        FilterName        = cms.untracked.InputTag("TauMCProducer","LeptonicTauElectrons"),
                                        matchObjectID     = cms.untracked.int32(11),
                                    ),
                                    cms.untracked.PSet(
                                        FilterName        = cms.untracked.InputTag("TauMCProducer","LeptonicTauMuons"),
                                        matchObjectID     = cms.untracked.int32(13),
                                    ),
                                    cms.untracked.PSet(
                                        FilterName        = cms.untracked.InputTag("TauMCProducer","MET"),
                                        matchObjectID     = cms.untracked.int32(0),
                                    ),
                                ),
    ),
)

hltTauValIdealMonitorPF = hltTauValIdealMonitorMC.clone(
    DQMBaseFolder = cms.untracked.string("HLT/TauRelVal/PF"),
    Matching = cms.PSet(
        doMatching            = cms.untracked.bool(True),
        matchFilters          = cms.untracked.VPSet(
                                    cms.untracked.PSet(
                                        FilterName        = cms.untracked.InputTag("TauRefCombiner",""),
                                        matchObjectID     = cms.untracked.int32(15),
                                    ),
                                    cms.untracked.PSet(
                                        FilterName        = cms.untracked.InputTag("TauMCProducer","LeptonicTauElectrons"),
                                        matchObjectID     = cms.untracked.int32(11),
                                    ),
                                    cms.untracked.PSet(
                                        FilterName        = cms.untracked.InputTag("TauMCProducer","LeptonicTauMuons"),
                                        matchObjectID     = cms.untracked.int32(13),
                                    ),
                                    cms.untracked.PSet(
                                        FilterName        = cms.untracked.InputTag("TauMCProducer","MET"),
                                        matchObjectID     = cms.untracked.int32(0),
                                    ),
                                ),
    ),
)

hltTauValIdeal = cms.Sequence(hltTauValIdealMonitorMC+hltTauValIdealMonitorPF)
