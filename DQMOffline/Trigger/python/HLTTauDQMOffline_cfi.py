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
                            ptMin = cms.untracked.double(10.0),
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
                            ptMin = cms.untracked.double(5.0)
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
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/PFTaus/DoubleLooseIsoTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleIsoTau20Trk5","",hltTauDQMofflineProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoublePFIsoTau20Trk5","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltDoublePFTauTightIso20Track","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltDoublePFTauTightIso20Track5","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterDoubleIsoPFTau20Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMofflineProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/PFTaus/EleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleEG12","",hltTauDQMofflineProcess), 
                                        cms.InputTag("hltEle15CaloIdVTTrkIdTCaloIsoTTrkIsoTTrackIsolFilter","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTau15","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTau15Track","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTau15TrackLooseIso","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltOverlapFilterIsoEle15IsoPFTau15","",hltTauDQMofflineProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84,84),
            LeptonType            = cms.untracked.vint32(-83,82,0,0,0,82)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/PFTaus/MuTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleMu10","",hltTauDQMofflineProcess), 
                                        cms.InputTag("hltL3Muon15","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTau20Track","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTau20TrackLooseIso","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltOverlapFilterMu15IsoPFTau20","",hltTauDQMofflineProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84),
            LeptonType            = cms.untracked.vint32(-81,83,0,0,83)                            
        ),
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/PFTaus/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterDoubleIsoPFTau20Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltOverlapFilterIsoEle15IsoPFTau15","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltOverlapFilterMu15IsoPFTau20","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20LeadTrackPt20","",hltTauDQMofflineProcess)
                                        ),
            PathName              = cms.untracked.vstring(
                                        'DoubleIsoTau',
                                        'EleTau',
                                        'MuTau',
                                        'SingleIsoTau'
                                        ),
            NTriggeredTaus        = cms.untracked.vuint32(2,1,1,1), 
            NTriggeredLeptons     = cms.untracked.vuint32(0,1,1,0), 
            TauType               = cms.untracked.vint32(15,15,15,15),
            LeptonType            = cms.untracked.vint32(0,11,13,0)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/PFTaus/SingleLooseIsoTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau35Trk20MET45","",hltTauDQMofflineProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoPFTau35Trk20MET45","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTauTightIso35","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTauTightIso35Track","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20LeadTrackPt20","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET45LeadTrack20MET45IsolationL1HLTMatched","",hltTauDQMofflineProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0,0)                         
        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOffline/PFTaus/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","NonIsolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles"),
        ),
        
   ),
    ConfigType = cms.vstring(
        "Path",
        "Path",
        "Path",
        "LitePath",
        "Path",
        "L1",
    ),
    
   doMatching = cms.bool(True),
      refObjects = cms.untracked.VInputTag(
      cms.InputTag("TauRefProducer","PFTaus"),
      cms.InputTag("TauRefProducer","Electrons"),
      cms.InputTag("TauRefProducer","Muons")
     )
)


hltTauOfflineMonitor_Inclusive = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/Inclusive/DoubleLooseIsoTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleIsoTau20Trk5","",hltTauDQMofflineProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoublePFIsoTau20Trk5","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltDoublePFTauTightIso20Track","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltDoublePFTauTightIso20Track5","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterDoubleIsoPFTau20Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMofflineProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/Inclusive/EleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleEG12","",hltTauDQMofflineProcess), 
                                        cms.InputTag("hltEle15CaloIdVTTrkIdTCaloIsoTTrkIsoTTrackIsolFilter","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTau15","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTau15Track","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTau15TrackLooseIso","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltOverlapFilterIsoEle15IsoPFTau15","",hltTauDQMofflineProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84,84),
            LeptonType            = cms.untracked.vint32(-83,82,0,0,0,82)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/Inclusive/MuTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleMu10","",hltTauDQMofflineProcess), 
                                        cms.InputTag("hltL3Muon15","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTau20Track","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTau20TrackLooseIso","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltOverlapFilterMu15IsoPFTau20","",hltTauDQMofflineProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84),
            LeptonType            = cms.untracked.vint32(-81,83,0,0,83)                            
        ),
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/Inclusive/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterDoubleIsoPFTau20Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltOverlapFilterIsoEle15IsoPFTau15","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltOverlapFilterMu15IsoPFTau20","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20LeadTrackPt20","",hltTauDQMofflineProcess)
                                        ),
            PathName              = cms.untracked.vstring(
                                        'DoubleIsoTau',
                                        'EleTau',
                                        'MuTau',
                                        'SingleIsoTau'
                                        ),
            NTriggeredTaus        = cms.untracked.vuint32(2,1,1,1), 
            NTriggeredLeptons     = cms.untracked.vuint32(0,1,1,0), 
            TauType               = cms.untracked.vint32(15,15,15,15),
            LeptonType            = cms.untracked.vint32(0,11,13,0)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/Inclusive/SingleLooseIsoTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau35Trk20MET45","",hltTauDQMofflineProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoPFTau35Trk20MET45","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTauTightIso35","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltPFTauTightIso35Track","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20LeadTrackPt20","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET45LeadTrack20MET45IsolationL1HLTMatched","",hltTauDQMofflineProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0,0)                         
        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOffline/Inclusive/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","NonIsolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles"),
        ),
        
   ),
    ConfigType = cms.vstring(
        "Path",
        "Path",
        "Path",
        "LitePath",
        "Path",
        "L1",
    ),
    
   doMatching = cms.bool(False),
      refObjects = cms.untracked.VInputTag(
      cms.InputTag("TauRefProducer","PFTaus"),
      cms.InputTag("TauRefProducer","Electrons"),
      cms.InputTag("TauRefProducer","Muons")
     )
)














