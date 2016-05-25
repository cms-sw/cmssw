import FWCore.ParameterSet.Config as cms

hltTauDQMofflineProcess = "HLT"

#Ref Objects-------------------------------------------------------------------------------------------------------
TauRefProducer = cms.EDProducer("HLTTauRefProducer",

                    PFTaus = cms.untracked.PSet(
                            PFTauDiscriminators = cms.untracked.VInputTag(
                                    cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),
                                    cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),
                                    cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection2")
                            ),
                            doPFTaus = cms.untracked.bool(True),
                            ptMin = cms.untracked.double(15.0),
                            PFTauProducer = cms.untracked.InputTag("hpsPFTauProducer")
                            ),
                    Electrons = cms.untracked.PSet(
                            ElectronCollection = cms.untracked.InputTag("gedGsfElectrons"),
                            doID = cms.untracked.bool(False),
                            InnerConeDR = cms.untracked.double(0.02),
                            MaxIsoVar = cms.untracked.double(0.02),
                            doElectrons = cms.untracked.bool(True),
                            TrackCollection = cms.untracked.InputTag("generalTracks"),
                            OuterConeDR = cms.untracked.double(0.6),
                            ptMin = cms.untracked.double(15.0),
                            doTrackIso = cms.untracked.bool(True),
                            ptMinTrack = cms.untracked.double(1.5),
                            lipMinTrack = cms.untracked.double(0.2),
                            IdCollection = cms.untracked.InputTag("elecIDext")
                            ),
                    Jets = cms.untracked.PSet(
                            JetCollection = cms.untracked.InputTag("ak4PFJetsCHS"),
                            etMin = cms.untracked.double(15.0),
                            doJets = cms.untracked.bool(False)
                            ),
                    Towers = cms.untracked.PSet(
                            TowerCollection = cms.untracked.InputTag("towerMaker"),
                            etMin = cms.untracked.double(10.0),
                            doTowers = cms.untracked.bool(False),
                            towerIsolation = cms.untracked.double(5.0)
                            ),

                    Muons = cms.untracked.PSet(
                            doMuons = cms.untracked.bool(True),
                            MuonCollection = cms.untracked.InputTag("muons"),
                            ptMin = cms.untracked.double(15.0)
                            ),

                    Photons = cms.untracked.PSet(
                            doPhotons = cms.untracked.bool(False),
                            PhotonCollection = cms.untracked.InputTag("gedPhotons"),
                            etMin = cms.untracked.double(15.0),
                            ECALIso = cms.untracked.double(3.0)
                            ),

                    MET = cms.untracked.PSet(
                            doMET = cms.untracked.bool(True),
                            METCollection = cms.untracked.InputTag("caloMet"), 
                            ptMin = cms.untracked.double(0.0)
                            ),

                    EtaMax = cms.untracked.double(2.3)
                  )

#----------------------------------MONITORS--------------------------------------------------------------------------

hltTauOfflineMonitor_PFTaus = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    HLTProcessName = cms.untracked.string(hltTauDQMofflineProcess),
    DQMBaseFolder = cms.untracked.string("HLT/TauOffline/PFTaus"),
    TriggerResultsSrc = cms.untracked.InputTag("TriggerResults", "", hltTauDQMofflineProcess),
    TriggerEventSrc = cms.untracked.InputTag("hltTriggerSummaryAOD", "", hltTauDQMofflineProcess),
    L1Plotter = cms.untracked.PSet(
        DQMFolder             = cms.untracked.string('L1'),
        L1Taus                = cms.untracked.InputTag("hltCaloStage2Digis", "Tau"),
        L1ETM                 = cms.untracked.InputTag("hltCaloStage2Digis","EtSum"),
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
                                    cms.untracked.PSet(
                                        FilterName        = cms.untracked.InputTag("TauRefProducer","MET"),
					matchObjectID     = cms.untracked.int32(0),
                                    ),
                                ),
    ),
)

hltTauOfflineMonitor_Inclusive = hltTauOfflineMonitor_PFTaus.clone(
    DQMBaseFolder = "HLT/TauOffline/Inclusive",
    Matching = cms.PSet(
        doMatching            = cms.untracked.bool(False),
        matchFilters          = cms.untracked.VPSet(),
    )
)
