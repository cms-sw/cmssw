import FWCore.ParameterSet.Config as cms
 
# import the whole HLT menu
#from HLTrigger.Configuration.HLT_8E29_cff import *
#from HLTrigger.Configuration.HLT_1E31_cff import *
from HLTrigger.Configuration.HLT_FULL_cff import *

# create the jetMET HLT reco path
DoHLTJets = cms.Path(HLTBeginSequence + 
    HLTBeginSequence +
    HLTRecoJetSequence +
    HLTRecoJetRegionalSequence +
    HLTRecoMETSequence +                 
    HLTDoJet15UHTRecoSequence
)
DoHLTJetsU = cms.Path(HLTBeginSequence +
    HLTBeginSequence +
    HLTRecoJetSequenceU +
    hltMet +
    HLTRecoJetRegionalSequence +
    HLTDoJet15UHTRecoSequence
)

# create the muon HLT reco path
DoHltMuon = cms.Path(
    HLTBeginSequence +
    HLTL2muonrecoSequence + 
    HLTL2muonisorecoSequence + 
    HLTL3muonrecoSequence + 
    HLTL3muonisorecoSequence +
    HLTMuTrackJpsiPixelRecoSequence + 
    HLTMuTrackJpsiTrackRecoSequence +
    HLTEndSequence )

# create the Egamma HLT reco paths
DoHLTPhoton = cms.Path( 
    HLTBeginSequence + 
    HLTDoRegionalEgammaEcalSequence + 
    HLTL1IsolatedEcalClustersSequence + 
    HLTL1NonIsolatedEcalClustersSequence + 
    hltL1IsoRecoEcalCandidate + 
    hltL1NonIsoRecoEcalCandidate + 
    HLTEgammaR9ShapeSequence +
    hltL1IsolatedPhotonEcalIsol + 
    hltL1NonIsolatedPhotonEcalIsol + 
    HLTDoLocalHcalWithoutHOSequence + 
    hltL1IsolatedPhotonHcalIsol + 
    hltL1NonIsolatedPhotonHcalIsol + 
#    HLTDoLocalTrackerSequence + 
    HLTL1IsoEgammaRegionalRecoTrackerSequence + 
    HLTL1NonIsoEgammaRegionalRecoTrackerSequence + 
    hltL1IsoPhotonHollowTrackIsol + 
    hltL1NonIsoPhotonHollowTrackIsol )

DoHLTElectron = cms.Path(
    HLTBeginSequence +
    HLTDoRegionalEgammaEcalSequence +
    HLTL1IsolatedEcalClustersSequence +
    HLTL1NonIsolatedEcalClustersSequence +
    hltL1IsoRecoEcalCandidate +
    hltL1NonIsoRecoEcalCandidate +
    HLTEgammaR9ShapeSequence +
    hltL1IsoHLTClusterShape +
    hltL1NonIsoHLTClusterShape +
    hltL1IsolatedPhotonEcalIsol +
    hltL1NonIsolatedPhotonEcalIsol +
    HLTDoLocalHcalWithoutHOSequence +
    hltL1IsolatedPhotonHcalForHE +
    hltL1NonIsolatedPhotonHcalForHE +
    hltL1IsolatedPhotonHcalIsol +
    hltL1NonIsolatedPhotonHcalIsol +
    HLTDoLocalPixelSequence +
    HLTDoLocalStripSequence +
    hltL1IsoStartUpElectronPixelSeeds +
    hltL1NonIsoStartUpElectronPixelSeeds +
    hltCkfL1IsoTrackCandidates +
    hltCtfL1IsoWithMaterialTracks +
    hltPixelMatchElectronsL1Iso +
    hltCkfL1NonIsoTrackCandidates +
    hltCtfL1NonIsoWithMaterialTracks +
    hltPixelMatchElectronsL1NonIso +
    hltElectronL1IsoDetaDphi +
    hltElectronL1NonIsoDetaDphi +
    HLTL1IsoEgammaRegionalRecoTrackerSequence +
    HLTL1NonIsoEgammaRegionalRecoTrackerSequence +
    hltL1IsoElectronTrackIsol + 
    hltL1NonIsoElectronTrackIsol
)

DoHLTElectronLargeWindows = cms.Path( 
    HLTBeginSequence  
    )

DoHLTElectronSiStrip = cms.Path( 
    HLTBeginSequence
    )

# create the tau HLT reco path
from HLTrigger.HLTanalyzers.OpenHLT_Tau_cff import *
DoHLTTau = cms.Path(HLTBeginSequence +
                    OpenHLTCaloTausCreatorSequence +
                    openhltL2TauJets +
                    openhltL2TauIsolationProducer +
#                    openhltL2TauRelaxingIsolationSelector +
                    HLTDoLocalPixelSequence +
                    HLTRecopixelvertexingSequence +
                    OpenHLTL25TauTrackReconstructionSequence +
                    OpenHLTL25TauTrackIsolation +
                    TauOpenHLT+
                    HLTEndSequence)


# create the b-jet HLT paths
from HLTrigger.HLTanalyzers.OpenHLT_BJet_cff import *
DoHLTBTag = cms.Path(
    HLTBeginSequence +
#    HLTBCommonL2recoSequence +
    OpenHLTBLifetimeL25recoSequence +
    OpenHLTBSoftMuonL25recoSequence +
    OpenHLTBLifetimeL3recoSequence +
    OpenHLTBLifetimeL3recoSequenceStartup +
    OpenHLTBSoftMuonL3recoSequence +
    HLTEndSequence )


DoHLTAlCaPi0Eta1E31 = cms.Path(
    HLTBeginSequence +
    hltL1sAlCaEcalPi0Eta1E31 +
    #hltPreAlCaEcalPi01E31 +
    #HLTDoRegionalPi0EtaESSequence +
    #HLTDoRegionalPi0EtaEcalSequence +
    HLTDoRegionalPi0EtaSequence +
    HLTEndSequence )

DoHLTAlCaPi0Eta8E29 = cms.Path(
    HLTBeginSequence +
    hltL1sAlCaEcalPi0Eta8E29 +
    #hltPreAlCaEcalPi08E29 +
    #HLTDoRegionalPi0EtaESSequence +
    #HLTDoRegionalPi0EtaEcalSequence +
    HLTDoRegionalPi0EtaSequence +
    HLTEndSequence )


DoHLTAlCaECALPhiSym = cms.Path(
    HLTBeginSequence +
    hltEcalRawToRecHitFacility + hltESRawToRecHitFacility + hltEcalRegionalRestFEDs + hltEcalRecHitAll +
##    hltAlCaPhiSymStream +
##    HLTDoLocalHcalSequence +
    HLTEndSequence )

#hltIsolPixelTrackProd1E31.MaxVtxDXYSeed = cms.double(101)
#hltIsolPixelTrackL2Filter1E31.MaxPtNearby = cms.double(3.0)
#hltIsolPixelTrackL2Filter1E31.MinPtTrack = cms.double(3.0)

#DoHLTIsoTrack = cms.Path(
#    HLTBeginSequence +
#    hltL1sIsoTrack1E31 +
    # hltPreIsoTrack1E31 +
#    HLTL2HcalIsolTrackSequence +
#    hltIsolPixelTrackProd1E31 +
#    hltIsolPixelTrackL2Filter1E31 +
#    HLTDoLocalStripSequence +
#    hltHITPixelPairSeedGenerator1E31 +
#    hltHITPixelTripletSeedGenerator1E31 +
#    hltHITSeedCombiner1E31 +
#    hltHITCkfTrackCandidates1E31 +
#    hltHITCtfWithMaterialTracks1E31 +
#    hltHITIPTCorrector1E31 +
#    HLTEndSequence)

#DoHLTIsoTrack8E29 = cms.Path(
#    HLTBeginSequence +
#    hltL1sIsoTrack8E29 +
    # hltPreIsoTrack8E29 +
#    HLTL2HcalIsolTrackSequence +
#    hltIsolPixelTrackProd8E29 +
#    hltIsolPixelTrackL2Filter8E29 +
#    HLTDoLocalStripSequence +
#    hltHITPixelPairSeedGenerator8E29 +
#    hltHITPixelTripletSeedGenerator8E29 +
#    hltHITSeedCombiner8E29 +
#    hltHITCkfTrackCandidates8E29 +
#    hltHITCtfWithMaterialTracks8E29 +
#    hltHITIPTCorrector8E29 +
#    HLTEndSequence)


DoHLTMinBiasPixelTracks = cms.Path(
    HLTBeginSequence +
    HLTDoLocalPixelSequence +
    HLTPixelTrackingForMinBiasSequence +
    hltPixelCandsForMinBias +
    hltPixelTracks +
    hltPixelVertices)

