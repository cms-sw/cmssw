import FWCore.ParameterSet.Config as cms
 
# import the whole HLT menu
from HLTrigger.HLTanalyzers.HLT_FULL_cff import *


# create the jetMET HLT reco path
DoHLTJets = cms.Path(HLTBeginSequence + 
    HLTBeginSequence +
    HLTRecoJetSequenceAK5Corrected +
    # HLTRegionalRecoJetSequenceAK5Corrected +
    HLTRecoMETSequence +                 
    HLTDoJet30HTRecoSequence
)
DoHLTJetsU = cms.Path(HLTBeginSequence +
    HLTBeginSequence +
    HLTRecoJetSequenceAK5Uncorrected +
    # HLTRegionalRecoJetSequenceAK5Corrected +
    HLTRecoMETSequence +
    HLTDoJet30HTRecoSequence
)

# create the muon HLT reco path
DoHltMuon = cms.Path(
    HLTBeginSequence +
    HLTL2muonrecoSequenceNoVtx +
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
    HLTEgammaR9IDSequence +
    hltL1IsolatedPhotonEcalIsol + 
    hltL1NonIsolatedPhotonEcalIsol + 
    HLTDoLocalHcalWithoutHOSequence + 
    hltL1IsolatedPhotonHcalIsol + 
    hltL1NonIsolatedPhotonHcalIsol + 
    HLTDoLocalPixelSequence +
    HLTDoLocalStripSequence +
    hltL1IsoEgammaRegionalPixelSeedGenerator +
    hltL1IsoEgammaRegionalCkfTrackCandidates +
    hltL1IsoEgammaRegionalCTFFinalFitWithMaterial +
    hltL1NonIsoEgammaRegionalPixelSeedGenerator +
    hltL1NonIsoEgammaRegionalCkfTrackCandidates +
    hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial +
    hltL1IsolatedPhotonHollowTrackIsol +
    hltL1NonIsolatedPhotonHollowTrackIsol )

DoHLTElectron = cms.Path(
    HLTBeginSequence +
    HLTDoRegionalEgammaEcalSequence +
    HLTL1IsolatedEcalClustersSequence +
    HLTL1NonIsolatedEcalClustersSequence +
    hltL1IsoRecoEcalCandidate +
    hltL1NonIsoRecoEcalCandidate +
    HLTEgammaR9ShapeSequence +
    HLTEgammaR9IDSequence +
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
    hltL1NonIsoElectronTrackIsol +
    hltHFEMClusters +
    hltHFRecoEcalCandidate
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
                    HLTRecoJetSequencePrePF +
                    HLTPFJetTriggerSequence +
                    HLTPFTauSequence +
                    HLTEndSequence)


# create the b-jet HLT paths
from HLTrigger.HLTanalyzers.OpenHLT_BJet_cff import *
# create the b-jet HLT paths
from HLTrigger.HLTanalyzers.OpenHLT_BJet_cff import *
DoHLTBTag = cms.Path(
        HLTBeginSequence +
    #    HLTBCommonL2recoSequence +
        OpenHLTBLifetimeL25recoSequence +
        OpenHLTBSoftMuonL25recoSequence +
        OpenHLTBLifetimeL3recoSequence +
        OpenHLTBSoftMuonL3recoSequence +
        HLTEndSequence )

DoHLTAlCaPi0Eta1E31 = cms.Path(
    HLTBeginSequence +
    hltL1sAlCaEcalPi0Eta +
    HLTDoRegionalPi0EtaSequence +
    HLTEndSequence )

DoHLTAlCaPi0Eta8E29 = cms.Path(
    HLTBeginSequence +
    hltL1sAlCaEcalPi0Eta +
    HLTDoRegionalPi0EtaSequence +
    HLTEndSequence )


DoHLTAlCaECALPhiSym = cms.Path(
    HLTBeginSequence +
    hltEcalRawToRecHitFacility + hltESRawToRecHitFacility + hltEcalRegionalRestFEDs + hltEcalRecHitAll +
    HLTEndSequence )


DoHLTMinBiasPixelTracks = cms.Path(
    HLTBeginSequence +
    # HLTDoLocalPixelSequence +
    HLTDoHILocalPixelSequence +
    HLTPixelTrackingForHITrackTrigger + 
    hltPixelCandsForHITrackTrigger +
    hltPixelTracks +
    hltPixelVertices)

## Thers is no need to do this as by default 5E32 menu makes use of "hltOnlineBeamSpot" for all the modules
# hltPixelVertices.beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
