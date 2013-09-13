import FWCore.ParameterSet.Config as cms

hltTauValidationProcess_IDEAL = "HLT"

hltTauValIdealMonitorMC = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    HLTProcessName = cms.untracked.string(hltTauValidationProcess_IDEAL),
    DQMBaseFolder = cms.untracked.string("HLT/TauRelVal/MC/"),
    TriggerResultsSrc = cms.untracked.InputTag("TriggerResults", "", hltTauValidationProcess_IDEAL),
    TriggerEventSrc = cms.untracked.InputTag("hltTriggerSummaryAOD", "", hltTauValidationProcess_IDEAL),
    MonitorSetup = cms.VPSet(
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            DQMFolder             = cms.untracked.string('TauMET'),
            Path                  = cms.untracked.vstring('HLT_LooseIsoPFTau35_Trk20_Prong1_MET(?<tr0>[[:digit:]]+)_v.*'),
            IgnoreFilterNames     = cms.untracked.vstring(),
            IgnoreFilterTypes     = cms.untracked.vstring(),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            DQMFolder             = cms.untracked.string('MuLooseTau'),
            Path                  = cms.untracked.vstring('HLT_IsoMu(?<tr1>[[:digit:]]+)_eta2p1_LooseIsoPFTau(?<tr0>[[:digit:]]+)_v.*'),
            IgnoreFilterNames     = cms.untracked.vstring(),
            IgnoreFilterTypes     = cms.untracked.vstring(),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            DQMFolder             = cms.untracked.string('MuMediumTau'),
            Path                  = cms.untracked.vstring('HLT_IsoMu(?<tr1>[[:digit:]]+)_eta2p1_MediumIsoPFTau(?<tr0>[[:digit:]]+)_Trk1_eta2p1_v.*'),
            IgnoreFilterNames     = cms.untracked.vstring(),
            IgnoreFilterTypes     = cms.untracked.vstring(),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            DQMFolder             = cms.untracked.string('EleTau'),
            Path                  = cms.untracked.vstring('HLT_Ele(?<tr1>[[:digit:]]+)_eta2p1_WP90Rho_LooseIsoPFTau(?<tr0>[[:digit:]]+)_v.*'),
            IgnoreFilterNames     = cms.untracked.vstring(),
            IgnoreFilterTypes     = cms.untracked.vstring(),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            DQMFolder             = cms.untracked.string('DoubleTau'),
            Path                  = cms.untracked.vstring('HLT_DoubleMediumIsoPFTau(?<tr0>[[:digit:]]+)_Trk(?<tr1>[[:digit:]])_eta2p1_Jet(?<tr2>[[:digit:]]+)_v.*'),
            IgnoreFilterNames     = cms.untracked.vstring(),
            IgnoreFilterTypes     = cms.untracked.vstring(),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("PathSummary"),
            DQMFolder             = cms.untracked.string('Summary'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("L1"),
            DQMFolder             = cms.untracked.string('L1'),
            L1Taus                = cms.untracked.InputTag("l1extraParticles", "Tau"),
            L1Jets                = cms.untracked.InputTag("l1extraParticles", "Central"),
            L1JetMinEt            = cms.untracked.double(40), # this value is arbitrary at the moment
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

hltTauValIdealMonitorPF = hltTauValIdealMonitorMC.clone(
    DQMBaseFolder = cms.untracked.string("HLT/TauRelVal/PF/"),
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
