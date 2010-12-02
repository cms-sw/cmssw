import FWCore.ParameterSet.Config as cms
 
# import the whole HLT menu
#from HLTrigger.Configuration.HLT_8E29_cff import *
#from HLTrigger.Configuration.HLT_1E31_cff import *
from HLTrigger.Configuration.HLT_FULL_cff import *

# create the jetMET HLT reco path
DoHLTJets = cms.Path(HLTBeginSequence + 
    HLTBeginSequence +
    HLTRecoJetSequenceU +
    HLTRecoJetRegionalSequence +
    HLTRecoMETSequence +                 
    HLTDoJet20UHTRecoSequence
)
DoHLTJetsU = cms.Path(HLTBeginSequence +
    HLTBeginSequence +
    HLTRecoJetSequenceU +
    hltMet +
    HLTRecoJetRegionalSequence +
    HLTDoJet20UHTRecoSequence
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
    hltL1sAlCaEcalPi0Eta8E29 +
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

DoHLTIsoTrackHE = cms.Path(
    HLTBeginSequence +
    hltL1sIsoTrackHE8E29 +
    hltPreIsoTrackHE8E29 +
    HLTL2HcalIsolTrackSequenceHE +
    hltIsolPixelTrackProdHE8E29 +
    hltIsolPixelTrackL2FilterHE8E29 +
    HLTDoLocalStripSequence +
    hltHITPixelTripletSeedGeneratorHE8E29 +
    hltHITCkfTrackCandidatesHE8E29 +
    hltHITCtfWithMaterialTracksHE8E29 +
    hltHITIPTCorrectorHE8E29 +
    hltIsolPixelTrackL3FilterHE8E29
    )

DoHLTIsoTrackHB = cms.Path(
    HLTBeginSequence +
    hltL1sIsoTrackHB8E29 +
    hltPreIsoTrackHB8E29 +
    HLTL2HcalIsolTrackSequenceHB +
    hltIsolPixelTrackProdHB8E29 +
    hltIsolPixelTrackL2FilterHB8E29 +
    HLTDoLocalStripSequence +
    hltHITPixelTripletSeedGeneratorHB8E29 +
    hltHITCkfTrackCandidatesHB8E29 +
    hltHITCtfWithMaterialTracksHB8E29 +
    hltHITIPTCorrectorHB8E29 +
    hltIsolPixelTrackL3FilterHB8E29
    )


DoHLTMinBiasPixelTracks = cms.Path(
    HLTBeginSequence +
    HLTDoLocalPixelSequence +
    HLTPixelTrackingForMinBiasSequence +
    hltPixelCandsForMinBias +
    hltPixelTracks +
    hltPixelVertices)

