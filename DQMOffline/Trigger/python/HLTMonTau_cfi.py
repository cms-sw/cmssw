import FWCore.ParameterSet.Config as cms

hltTauDQMProcess = "HLT"

hltTauMonitor = cms.EDAnalyzer("HLTTauDQMSource",
    HLTProcessName = cms.untracked.string(hltTauDQMProcess),
    ModuleName = cms.untracked.string("hltTauMonitor"),
    DQMBaseFolder = cms.untracked.string("HLT/TauOnline/Inclusive/"),
    MonitorSetup = cms.VPSet(
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('DoubleTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('Ele.+?Tau'),
            Alias                 = cms.untracked.string('EleTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('Mu.+?Tau'),
            Alias                 = cms.untracked.string('MuTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('Single.+?Tau_MET'),
            Alias                 = cms.untracked.string('SingleTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("LitePath"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('Summary'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("L1"),
            DQMFolder             = cms.untracked.string('L1'),
            L1Taus                = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons           = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons               = cms.InputTag("hltL1extraParticles"),
        ),
    ),
    Matching = cms.PSet(
        doMatching            = cms.untracked.bool(False),
        TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
        matchFilters          = cms.untracked.VPSet(),
    ),
)

hltTauElectronMonitor = cms.EDAnalyzer("HLTTauDQMSource",
    HLTProcessName = cms.untracked.string(hltTauDQMProcess),
    ModuleName = cms.untracked.string("hltTauElectronMonitor"),
    DQMBaseFolder = cms.untracked.string("HLT/TauOnline/Electrons/"),
    MonitorSetup = cms.VPSet(
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('DoubleTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('Single.+?Tau_MET'),
            Alias                 = cms.untracked.string('SingleTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('LoosePFTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('MediumPFTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('TightPFTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("LitePath"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('Summary'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("L1"),
            DQMFolder             = cms.untracked.string('L1'),
            L1Taus                = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons           = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons               = cms.InputTag("hltL1extraParticles"),
        ),
    ),
    Matching = cms.PSet(
        doMatching            = cms.untracked.bool(True),
        TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
        matchFilters          = cms.untracked.VPSet(
                                    cms.untracked.PSet(
                                        AutomaticFilterName   = cms.untracked.string('Ele.+?Tau'),
                                        matchObjectID         = cms.untracked.int32(11),
                                        matchObjectMinPt      = cms.untracked.double(10),
                                    ),
                                ),
    ),
)

hltMonTauReco =cms.Sequence(hltTauMonitor+hltTauElectronMonitor)

