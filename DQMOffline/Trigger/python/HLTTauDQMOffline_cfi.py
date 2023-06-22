import FWCore.ParameterSet.Config as cms

hltTauDQMofflineProcess = "HLT"

#Ref Objects-------------------------------------------------------------------------------------------------------
TauRefProducer = cms.EDProducer("HLTTauRefProducer",

                    PFTaus = cms.untracked.PSet(
                            PFTauDiscriminatorContainers  = cms.untracked.VInputTag(),
                            PFTauDiscriminatorContainerWPs  = cms.untracked.vstring(),
                            PFTauDiscriminators = cms.untracked.VInputTag(
                                    cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding")
                                    #cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),
                                    #cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection3"),
                                    #cms.InputTag("hpsPFTauDiscriminationByMVA6TightElectronRejection")
                            ),
                            doPFTaus = cms.untracked.bool(True),
                            ptMin = cms.untracked.double(15.0),
                            etaMin = cms.untracked.double(-2.5),
                            etaMax = cms.untracked.double(2.5),
                            phiMin = cms.untracked.double(-3.15),
                            phiMax = cms.untracked.double(3.15),
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

                    EtaMin = cms.untracked.double(-2.3),
                    EtaMax = cms.untracked.double(2.3),
                    PhiMin = cms.untracked.double(-3.15),
                    PhiMax = cms.untracked.double(3.15)
                  )

#----------------------------------MONITORS--------------------------------------------------------------------------
kEverything = 0
kVital      = 1

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hltTauOfflineMonitor_PFTaus = DQMEDAnalyzer('HLTTauDQMOfflineSource',
    HLTProcessName = cms.untracked.string(hltTauDQMofflineProcess),
    DQMBaseFolder = cms.untracked.string("HLT/TAU/PFTaus"),
    PlotLevel = cms.untracked.int32(kVital),
    TriggerResultsSrc = cms.untracked.InputTag("TriggerResults", "", hltTauDQMofflineProcess),
    TriggerEventSrc = cms.untracked.InputTag("hltTriggerSummaryAOD", "", hltTauDQMofflineProcess),
    L1Plotter = cms.untracked.PSet(
        DQMFolder             = cms.untracked.string('L1'),
        L1Taus                = cms.untracked.InputTag("caloStage2Digis", "Tau"),
        L1ETM                 = cms.untracked.InputTag("caloStage2Digis","EtSum"),
        L1ETMMin              = cms.untracked.double(50),
    ),
    Paths = cms.untracked.string("PFTau"),
    PtHistoBins = cms.untracked.int32(50),
    PtHistoMax = cms.untracked.double(500),
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
    DQMBaseFolder = "HLT/TAU/Inclusive",
    Matching = cms.PSet(
        doMatching            = cms.untracked.bool(False),
        matchFilters          = cms.untracked.VPSet(),
    )
)

def TriggerSelectionParameters(hltpaths):
    genericTriggerSelectionParameters = cms.PSet(
                andOr          = cms.bool( False ),#specifies the logical combination of the single filters' (L1, HLT and DCS) decisions at top level (True=OR)
                dbLabel        = cms.string("PFTauDQMTrigger"),#specifies the label under which the DB payload is available from the ESSource or Global Tag
                andOrHlt       = cms.bool(True),#specifies the logical combination of the single HLT paths' decisions (True=OR)
                hltInputTag    = cms.InputTag("TriggerResults", "", hltTauDQMofflineProcess),
                hltPaths       = hltpaths,#Lists logical expressions of HLT paths, which should have accepted the event (fallback in case DB unaccessible)
                errorReplyHlt  = cms.bool(False),#specifies the desired return value of the HLT filter and the single HLT path filter in case of certain errors
                verbosityLevel = cms.uint32(0) #0: complete silence (default), needed for T0 processing;
    )
    return genericTriggerSelectionParameters


hltTauOfflineMonitor_TagAndProbe = hltTauOfflineMonitor_PFTaus.clone(
    DQMBaseFolder = "HLT/TAU/TagAndProbe",
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
    TagAndProbe = cms.untracked.VPSet(
        cms.untracked.PSet(
            name        = cms.string('MuTauTemplate'),
            xvariable   = cms.string('Tau'),
            nPtBins     = cms.int32(20),
            ptmin       = cms.double(0.),
            ptmax       = cms.double(200.),
            nEtaBins    = cms.int32(20),  
            etamin      = cms.double(-2.5),
            etamax      = cms.double(2.5),
            nPhiBins    = cms.int32(20),  
            phimin      = cms.double(-3.15),
            phimax      = cms.double(3.15),
            numerator   = TriggerSelectionParameters(cms.vstring('HLT_IsoMu24_eta2p1_.+PFTau.+')),
            denominator = TriggerSelectionParameters(cms.vstring('HLT_IsoMu27_v*'))
        ),
        cms.untracked.PSet(
            name        = cms.string('ETauTemplate'),
            xvariable   = cms.string('Tau'),
            nPtBins     = cms.int32(20),
            ptmin       = cms.double(0.),  
            ptmax       = cms.double(200.),
            nEtaBins    = cms.int32(20),   
            etamin      = cms.double(-2.5),
            etamax      = cms.double(2.5), 
            nPhiBins    = cms.int32(20),  
            phimin      = cms.double(-3.15),
            phimax      = cms.double(3.15), 
            numerator   = TriggerSelectionParameters(cms.vstring('HLT_Ele.+PFTau.+')),
            denominator = TriggerSelectionParameters(cms.vstring('HLT_Ele35_WPTight_Gsf_v*'))
        ),
        cms.untracked.PSet(
            name        = cms.string('TauMETTemplate'),
            xvariable   = cms.string('MET'),
            nPtBins     = cms.int32(50),
            ptmin       = cms.double(0.),
            ptmax       = cms.double(500.),
            nPhiBins    = cms.int32(20),
            phimin      = cms.double(-3.15),
            phimax      = cms.double(3.15),
            numerator   = TriggerSelectionParameters(cms.vstring('HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET.*')),
            denominator = TriggerSelectionParameters(cms.vstring('HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_v*'))
        ),
        cms.untracked.PSet(
            name        = cms.string('IsoMu20_eta2p1'),
            xvariable   = cms.string('Muon'),
            nPtBins     = cms.int32(20),
            ptmin       = cms.double(0.),
            ptmax       = cms.double(200.),
            nEtaBins    = cms.int32(20),
            etamin      = cms.double(-2.5),  
            etamax      = cms.double(2.5),
            nPhiBins    = cms.int32(20),
            phimin      = cms.double(-3.15),
            phimax      = cms.double(3.15),
            numerator   = TriggerSelectionParameters(cms.vstring('HLT_DoubleIsoMu20_eta2p1_v*')),
            denominator = TriggerSelectionParameters(cms.vstring('HLT_IsoMu27_v*')),
            nOfflObjs   = cms.untracked.uint32(2)
        ),
        cms.untracked.PSet(
            name        = cms.string('IsoMu24_eta2p1'),
            xvariable   = cms.string('Muon'),
            nPtBins     = cms.int32(20),  
            ptmin       = cms.double(0.),  
            ptmax       = cms.double(200.),
            nEtaBins    = cms.int32(20),
            etamin      = cms.double(-2.5),  
            etamax      = cms.double(2.5), 
            nPhiBins    = cms.int32(20),   
            phimin      = cms.double(-3.15),
            phimax      = cms.double(3.15),
            numerator   = TriggerSelectionParameters(cms.vstring('HLT_DoubleIsoMu24_eta2p1_v*')),
            denominator = TriggerSelectionParameters(cms.vstring('HLT_IsoMu27_v*'))
        ),
        cms.untracked.PSet(
            name        = cms.string('Ele24_eta2p1_WPTight_Gsf'),
            xvariable   = cms.string('Electron'),
            nPtBins     = cms.int32(20),   
            ptmin       = cms.double(0.),
            ptmax       = cms.double(200.),
            nEtaBins    = cms.int32(20),
            etamin      = cms.double(-2.5),  
            etamax      = cms.double(2.5), 
            nPhiBins    = cms.int32(20),   
            phimin      = cms.double(-3.15),
            phimax      = cms.double(3.15),
            numerator   = TriggerSelectionParameters(cms.vstring('HLT_DoubleEle24_eta2p1_WPTight_Gsf_v*')),
            denominator = TriggerSelectionParameters(cms.vstring('HLT_Ele35_WPTight_Gsf_v*'))
        )
    )
)
