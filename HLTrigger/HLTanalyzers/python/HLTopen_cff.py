import FWCore.ParameterSet.Config as cms
 
# import the whole HLT menu
from HLTrigger.Configuration.HLT_8E29_cff import *

# create the muon HLT reco path
DoHltMuon = cms.Path(
    HLTBeginSequence +
    HLTL2muonrecoSequence + 
    HLTL2muonisorecoSequence + 
    HLTL3muonrecoSequence + 
    HLTL3muonisorecoSequence +
    HLTEndSequence )

# create the Egamma HLT reco paths
DoHLTPhoton = cms.Path( 
    HLTBeginSequence + 
    HLTDoRegionalEgammaEcalSequence + 
    HLTL1IsolatedEcalClustersSequence + 
    HLTL1NonIsolatedEcalClustersSequence + 
    hltL1IsoRecoEcalCandidate + 
    hltL1NonIsoRecoEcalCandidate + 
    hltL1IsolatedPhotonEcalIsol + 
    hltL1NonIsolatedPhotonEcalIsol + 
    HLTDoLocalHcalWithoutHOSequence + 
    hltL1IsolatedPhotonHcalIsol + 
    hltL1NonIsolatedPhotonHcalIsol + 
    HLTDoLocalTrackerSequence + 
    HLTL1IsoEgammaRegionalRecoTrackerSequence + 
    HLTL1NonIsoEgammaRegionalRecoTrackerSequence + 
    hltL1IsoPhotonTrackIsol + 
    hltL1NonIsoPhotonTrackIsol )

DoHLTElectron = cms.Path( 
    HLTBeginSequence + 
    HLTDoRegionalEgammaEcalSequence + 
    HLTL1IsolatedEcalClustersSequence + 
    HLTL1NonIsolatedEcalClustersSequence + 
    hltL1IsoRecoEcalCandidate + 
    hltL1NonIsoRecoEcalCandidate + 
    HLTDoLocalHcalWithoutHOSequence + 
    hltL1IsolatedElectronHcalIsol + 
    hltL1NonIsolatedElectronHcalIsol + 
    HLTDoLocalPixelSequence + 
    HLTDoLocalStripSequence + 
    HLTPixelMatchElectronL1IsoSequence + 
    HLTPixelMatchElectronL1NonIsoSequence + 
    HLTPixelMatchElectronL1IsoTrackingSequence + 
    HLTPixelMatchElectronL1NonIsoTrackingSequence + 
    HLTL1IsoElectronsRegionalRecoTrackerSequence + 
    HLTL1NonIsoElectronsRegionalRecoTrackerSequence + 
    hltL1IsoElectronTrackIsol + 
    hltL1NonIsoElectronTrackIsol )

DoHLTElectronStartUpWindows = cms.Path( 
    HLTBeginSequence + 
    HLTDoRegionalEgammaEcalSequence + 
    HLTL1IsolatedEcalClustersSequence + 
    HLTL1NonIsolatedEcalClustersSequence + 
    hltL1IsoRecoEcalCandidate + 
    hltL1NonIsoRecoEcalCandidate + 
    HLTDoLocalHcalWithoutHOSequence + 
    hltL1IsolatedElectronHcalIsol + 
    hltL1NonIsolatedElectronHcalIsol + 
    HLTDoLocalPixelSequence + 
    HLTDoLocalStripSequence + 
    hltL1IsoStartUpElectronPixelSeeds + 
    hltL1NonIsoStartUpElectronPixelSeeds + 
    HLTPixelMatchStartUpElectronL1IsoTrackingSequence + 
    HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + 
    HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + 
    HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + 
    hltL1IsoStartUpElectronTrackIsol + 
    hltL1NonIsoStartupElectronTrackIsol )

DoHLTElectronLargeWindows = cms.Path( 
    HLTBeginSequence + 
    HLTDoRegionalEgammaEcalSequence + 
    HLTL1IsolatedEcalClustersSequence + 
    HLTL1NonIsolatedEcalClustersSequence + 
    hltL1IsoRecoEcalCandidate + 
    hltL1NonIsoRecoEcalCandidate + 
    HLTDoLocalHcalWithoutHOSequence + 
    hltL1IsolatedElectronHcalIsol + 
    hltL1NonIsolatedElectronHcalIsol + 
    HLTDoLocalPixelSequence + 
    HLTDoLocalStripSequence + 
    HLTPixelMatchElectronL1IsoLargeWindowSequence + 
    HLTPixelMatchElectronL1NonIsoLargeWindowSequence + 
    HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + 
    HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + 
    HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + 
    HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + 
    hltL1IsoLargeWindowElectronTrackIsol + 
    hltL1NonIsoLargeWindowElectronTrackIsol )

# create the jetMET HLT reco path
DoHLTJets = cms.Path(HLTBeginSequence + 
    HLTRecoJetMETSequence + 
    HLTDoHTRecoSequence)

# create the tau HLT reco path
from HLTrigger.HLTanalyzers.OpenHLT_Tau_cff import *
DoHLTTau = cms.Path(HLTBeginSequence +
                    hltTauPrescaler +
                    OpenHLTCaloTausCreatorSequence +
                    hltL2TauJets +
                    hltL2TauIsolationProducer +
                    hltL2TauRelaxingIsolationSelector +
                    HLTDoLocalPixelSequence +
                    HLTRecopixelvertexingSequence +
                    HLTL25TauTrackReconstructionSequence +
                    HLTL25TauTrackIsolation +
                    TauOpenHLT+
                    HLTEndSequence)

# create the b-jet HLT paths
from HLTrigger.HLTanalyzers.OpenHLT_BJet_cff import *
DoHLTBTag = cms.Path(
    HLTBeginSequence +
    HLTBCommonL2recoSequence +
    OpenHLTBLifetimeL25recoSequence +
    OpenHLTBSoftmuonL25recoSequence +
    OpenHLTBLifetimeL3recoSequence +
    OpenHLTBLifetimeL3recoSequenceRelaxed +
    OpenHLTBSoftmuonL3recoSequence +
    HLTEndSequence )


# create special ECAL rechits for the AlCa Pi0 path.
hltEcalRegionalPi0FEDs = cms.EDProducer( "EcalListOfFEDSProducer",
                                         debug = cms.untracked.bool( False ),
                                         Pi0ListToIgnore =  cms.InputTag("hltEcalRegionalPi0FEDs"),                                              \
                                         EGamma = cms.untracked.bool( True ),
                                         EM_l1TagIsolated = cms.untracked.InputTag( 'hltL1extraParticles','Isolated' ),
                                         EM_l1TagNonIsolated = cms.untracked.InputTag( 'hltL1extraParticles','NonIsolated' ),
                                         Ptmin_iso = cms.untracked.double( 2.0 ),
                                         Ptmin_noniso = cms.untracked.double( 2.0 ),
                                         OutputLabel = cms.untracked.string( "" )
                                         )

hltEcalRegionalPi0Digis = cms.EDProducer( "EcalRawToDigi",
                                          syncCheck = cms.untracked.bool( False ),
                                          eventPut = cms.untracked.bool( True ),
                                          InputLabel = cms.untracked.string( "rawDataCollector" ),
                                          DoRegional = cms.untracked.bool( True ),
                                          FedLabel = cms.untracked.InputTag( "hltEcalRegionalPi0FEDs" ),
                                          silentMode = cms.untracked.bool( True ),
                                          orderedFedList = cms.untracked.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
                                          orderedDCCIdList = cms.untracked.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 )
                                          )

hltEcalRegionalPi0WeightUncalibRecHit = cms.EDProducer( "EcalUncalibRecHitProducer",
                                                        EBdigiCollection = cms.InputTag( 'hltEcalRegionalPi0Digis','ebDigis' ),
                                                        EEdigiCollection = cms.InputTag( 'hltEcalRegionalPi0Digis','eeDigis' ),
                                                        EBhitCollection = cms.string( "EcalUncalibRecHitsEB" ),
                                                        EEhitCollection = cms.string( "EcalUncalibRecHitsEE" ),
                                                        algo = cms.string("EcalUncalibRecHitWorkerWeights")
                                                        )

hltEcalRegionalPi0RecHitTmp = cms.EDProducer( "EcalRecHitProducer",
                                              EBuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalPi0WeightUncalibRecHit','EcalUncalibRecHitsEB'),
                                              EEuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalPi0WeightUncalibRecHit','EcalUncalibRecHitsEE'),
                                              EBrechitCollection = cms.string( "EcalRecHitsEB" ),
                                              EErechitCollection = cms.string( "EcalRecHitsEE" ),
                                              ChannelStatusToBeExcluded = cms.vint32(  ),
                                              algo = cms.string("EcalRecHitWorkerSimple")
                                              )

hltEcalRegionalPi0RecHit = cms.EDProducer( "EcalRecHitsMerger",
                                           debug = cms.untracked.bool( False ),
                                           EgammaSource_EB = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEB' ),
                                           MuonsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEB' ),
                                           TausSource_EB = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEB' ),
                                           JetsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEB' ),
                                           RestSource_EB = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEB' ),
                                           Pi0Source_EB =  cms.untracked.InputTag( 'hltEcalRegionalPi0RecHitTmp','EcalRecHitsEB'),
                                           EgammaSource_EE = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEE' ),
                                           MuonsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEE' ),
                                           TausSource_EE = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEE' ),
                                           JetsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEE' ),
                                           RestSource_EE = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEE' ),
                                           Pi0Source_EE =  cms.untracked.InputTag( 'hltEcalRegionalPi0RecHitTmp','EcalRecHitsEE' ),
                                           OutputLabel_EB = cms.untracked.string( "EcalRecHitsEB" ),
                                           OutputLabel_EE = cms.untracked.string( "EcalRecHitsEE" ),
                                           EcalRecHitCollectionEB = cms.untracked.string( "EcalRecHitsEB" ),
                                           EcalRecHitCollectionEE = cms.untracked.string( "EcalRecHitsEE" )
                                           )

HLTDoRegionalPi0EcalSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRegionalPi0FEDs + hltEcalRegionalPi0Digis + hltEcalRegionalPi0WeightUncalibRecHit + hltEcalRegionalPi0RecHitTmp + hltEcalRegionalPi0RecHit + hltEcalPreshowerRecHit )

DoHLTAlCaPi0 = cms.Path(
    HLTDoRegionalPi0EcalSequence
    )

DoHLTAlCaECALPhiSym = cms.Path(
    HLTBeginSequence +
    # ccla hltL1sAlCaEcalPhiSym +
    hltPreAlCaEcalPhiSym +
    hltEcalDigis +
    hltEcalWeightUncalibRecHit +
    hltEcalRecHit +
    hltAlCaPhiSymStream +
    HLTDoLocalHcalSequence +
    HLTEndSequence
    )
