import FWCore.ParameterSet.Config as cms

hltTauValidationProcess_IDEAL = "HLT"

hltTauValIdealMonitorMC = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    ModuleName = cms.untracked.string("hltTauValIdealMonitorMC"),
    DQMBaseFolder = cms.untracked.string("HLT/TauRelVal/MC/"),
    HLTProcessName = cms.untracked.string(hltTauValidationProcess_IDEAL),
    MonitorSetup = cms.VPSet(
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string('DoubleTau'),                        
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string('Ele.+?Tau'),                        
            Alias                 = cms.untracked.string('EleTau'),                        
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string('MuLooseTau'),                           
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string('MuMediumTau'),                          
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string('MuTightTau'),                         
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string('Single.+?Tau_MET'),
            Alias                 = cms.untracked.string('SingleTau'),                        
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("LitePath"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string('Summary'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("L1"),
            DQMFolder             = cms.untracked.string('L1'),
            L1Taus                = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons           = cms.InputTag("hltL1extraParticles","NonIsolated"),
            L1Muons               = cms.InputTag("hltL1extraParticles"),
        ),
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
                                ),
    ),
)

hltTauValIdealMonitorPF = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    ModuleName = cms.untracked.string("hltTauValIdealMonitorPF"),
    DQMBaseFolder = cms.untracked.string("HLT/TauRelVal/PF/"),
    HLTProcessName = cms.untracked.string(hltTauValidationProcess_IDEAL),
    MonitorSetup = hltTauValIdealMonitorMC.MonitorSetup,
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
                                ),
    ),
)

hltTauValIdeal = cms.Sequence(hltTauValIdealMonitorMC+hltTauValIdealMonitorPF)
