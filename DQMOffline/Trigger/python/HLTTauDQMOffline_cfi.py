import FWCore.ParameterSet.Config as cms

hltTauDQMofflineProcess = "HLT2"


#Ref Pbjects-------------------------------------------------------------------------------------------------------
TauRefProducer = cms.EDFilter("HLTTauRefProducer",

                    PFTaus = cms.untracked.PSet(
                            PFTauDiscriminator = cms.untracked.InputTag("pfRecoTauDiscriminationByIsolation"),
                            doPFTaus = cms.untracked.bool(True),
                            ptMin = cms.untracked.double(10.0),
                            PFTauProducer = cms.untracked.InputTag("pfRecoTauProducer")
                            ),
                    CaloTaus = cms.untracked.PSet(
                            ptMinTau = cms.untracked.double(10.0),
                            doCaloTaus = cms.untracked.bool(True),
                            CaloTauProducer = cms.untracked.InputTag("caloRecoTauProducer"),
                            CaloTauDiscriminator = cms.untracked.InputTag("caloRecoTauDiscriminationByIsolation")
                            ),

                    Electrons = cms.untracked.PSet(
                            ElectronCollection = cms.untracked.InputTag("pixelMatchGsfElectrons"),
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
                   Muons = cms.untracked.PSet(
                            doMuons = cms.untracked.bool(True),
                            MuonCollection = cms.untracked.InputTag("muons"),
                            ptMin = cms.untracked.double(5.0)
                            ),

                  EtaMax = cms.untracked.double(2.5)
                  )

#----------------------------------MONITORS--------------------------------------------------------------------------



hltTauOfflineMonitor_PFTaus = cms.EDFilter("HLTTauDQMOfflineSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/PFTaus/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleLooseIsoTau15Trk5","",hltTauDQMofflineProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoubleLooseIsoTau15Trk5","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15Trk5","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutDoubleLooseIsoTau15Trk5","",hltTauDQMofflineProcess),
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3,0.3),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(86,94,94,94),
            LeptonType            = cms.untracked.vint32(0,0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/PFTaus/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutDoubleLooseIsoTau15Trk5","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterL3IsolationCutSingleIsoTau20Trk5","",hltTauDQMofflineProcess)
                                        ),
            PathName              = cms.untracked.vstring(
                                        'DoubleLooseIsoTau',
                                        'SingleLooseIsoTau'
                                        ),
            NTriggeredTaus        = cms.untracked.vuint32(2,1), 
            NTriggeredLeptons     = cms.untracked.vuint32(0,0), 
            TauType               = cms.untracked.vint32(15,15),
            LeptonType            = cms.untracked.vint32(0,0)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMofflineProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOffline/PFTaus/SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau20Trk5","",hltTauDQMofflineProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoTau20Trk5","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleIsoTau20Trk5","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutSingleIsoTau20Trk5","",hltTauDQMofflineProcess),
                                        cms.InputTag("hltFilterL3IsolationCutSingleIsoTau20Trk5","",hltTauDQMofflineProcess),
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3,0.3,0,0.3),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(86,94,94,94,94),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            

        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOffline/PFTaus/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Leptons              = cms.InputTag("none"),
            LeptonType             = cms.int32(0),
            NTriggeredTaus         = cms.uint32(2),
            NTriggeredLeptons      = cms.uint32(0)
        ),

        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOffline/PFTaus/L2'),
            L2InfoAssociationInput = cms.InputTag("hltL2TauIsolationProducer"),
            L2IsolatedJets         = cms.InputTag("hltL2TauRelaxingIsolationSelector","Isolated")
        ),


        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOffline/PFTaus/L25'),
            ConeIsolation          = cms.InputTag("hltL25TauConeIsolation"),
            IsolatedJets           = cms.InputTag("hltL25TauLeadingTrackPtCutSelector"),                             
            Type                   = cms.string('L25')                           
        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOffline/PFTaus/L3'),
            ConeIsolation          = cms.InputTag("hltL3TauConeIsolation"),
            IsolatedJets           = cms.InputTag("hltL3TauIsolationSelector"),                             
            Type                   = cms.string('L3')                           
        )
   ),
    ConfigType = cms.vstring(
        "Path",
        "LitePath",
        "Path",
        "L1",
        "Calo",
        "Track",
        "Track"
    ),
    
   doMatching = cms.bool(True),
      refObjects = cms.untracked.VInputTag(
      cms.InputTag("TauRefProducer","PFTaus",hltTauDQMofflineProcess)
     )
)













