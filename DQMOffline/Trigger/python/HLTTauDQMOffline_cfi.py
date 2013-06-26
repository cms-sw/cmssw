import FWCore.ParameterSet.Config as cms

hltTauDQMofflineProcess = "HLT"

#Ref Pbjects-------------------------------------------------------------------------------------------------------
TauRefProducer = cms.EDProducer("HLTTauRefProducer",

                    PFTaus = cms.untracked.PSet(
                            PFTauDiscriminators = cms.untracked.VInputTag(
                            						cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),
                                                    cms.InputTag("hpsPFTauDiscriminationByLooseIsolation")
                            ),
                            doPFTaus = cms.untracked.bool(True),
                            ptMin = cms.untracked.double(15.0),
                            PFTauProducer = cms.untracked.InputTag("hpsPFTauProducer")
                            ),
                    Electrons = cms.untracked.PSet(
                            ElectronCollection = cms.untracked.InputTag("gsfElectrons"),
                            doID = cms.untracked.bool(False),
                            InnerConeDR = cms.untracked.double(0.02),
                            MaxIsoVar = cms.untracked.double(0.02),
                            doElectrons = cms.untracked.bool(True),
                            TrackCollection = cms.untracked.InputTag("generalTracks"),
                            OuterConeDR = cms.untracked.double(0.6),
                            ptMin = cms.untracked.double(10.0),
                            doTrackIso = cms.untracked.bool(True),
                            ptMinTrack = cms.untracked.double(1.5),
                            lipMinTrack = cms.untracked.double(0.2),
                            IdCollection = cms.untracked.InputTag("elecIDext")
                            ),
                   Jets = cms.untracked.PSet(
                            JetCollection = cms.untracked.InputTag("iterativeCone5CaloJets"),
                            etMin = cms.untracked.double(10.0),
                            doJets = cms.untracked.bool(True)
                            ),
                   Towers = cms.untracked.PSet(
                            TowerCollection = cms.untracked.InputTag("towerMaker"),
                            etMin = cms.untracked.double(10.0),
                            doTowers = cms.untracked.bool(True),
                            towerIsolation = cms.untracked.double(5.0)
                            ),

                   Muons = cms.untracked.PSet(
                            doMuons = cms.untracked.bool(True),
                            MuonCollection = cms.untracked.InputTag("muons"),
                            ptMin = cms.untracked.double(10.0)
                            ),

                   Photons = cms.untracked.PSet(
                            doPhotons = cms.untracked.bool(True),
                            PhotonCollection = cms.untracked.InputTag("photons"),
                            etMin = cms.untracked.double(10.0),
                            ECALIso = cms.untracked.double(3.0)
                            ),
                  EtaMax = cms.untracked.double(2.5)
                  )

#----------------------------------MONITORS--------------------------------------------------------------------------

hltTauOfflineMonitor_PFTaus = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    HLTProcessName = cms.untracked.string(hltTauDQMofflineProcess),
    ModuleName = cms.untracked.string("hltTauOfflineMonitor_PFTaus"),
    DQMBaseFolder = cms.untracked.string("HLT/TauOffline/PFTaus/"),
    MonitorSetup = cms.VPSet(
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('DoubleTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('Ele.+?Tau'),
            Alias                 = cms.untracked.string('EleTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('MuLooseTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('MuMediumTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('MuTightTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('Single.+?Tau_MET'),
            Alias                 = cms.untracked.string('SingleTau'),
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("LitePath"),
            TriggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
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
                                        FilterName        = cms.untracked.InputTag("TauRefProducer","PFTaus"),
                                        matchObjectID     = cms.untracked.int32(15),
                                    ),
                                    cms.untracked.PSet(
                                        FilterName        = cms.untracked.InputTag("TauRefProducer","Electrons"),
                                        matchObjectID     = cms.untracked.int32(11),
                                    ),
                                    cms.untracked.PSet(
                                        FilterName        = cms.untracked.InputTag("TauRefProducer","Muons"),
                                        matchObjectID     = cms.untracked.int32(13),
                                    ),
                                ),
    ),
)

hltTauOfflineMonitor_Inclusive = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    HLTProcessName = cms.untracked.string(hltTauDQMofflineProcess),
    ModuleName = cms.untracked.string("hltTauOfflineMonitor_Inclusive"),
    DQMBaseFolder = cms.untracked.string("HLT/TauOffline/Inclusive/"),
    MonitorSetup = hltTauOfflineMonitor_PFTaus.MonitorSetup,
    Matching = cms.PSet(
        doMatching            = cms.untracked.bool(False),
        matchFilters          = cms.untracked.VPSet(),
    ),
)
