import FWCore.ParameterSet.Config as cms
 
# import the whole HLT menu
#from HLTrigger.Configuration.HLT_8E29_cff import *
#from HLTrigger.Configuration.HLT_1E31_cff import *
from HLTrigger.Configuration.HLT_FULL_cff import *


#
# Temporarily add EGamma R9-ID here until it goes in the full HLT configuration
# 
hltL1IsoR9ID = cms.EDProducer( "EgammaHLTR9IDProducer",
                                          recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
                                          ecalRechitEB = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
                                          ecalRechitEE = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' )
                               )
hltL1NonIsoR9ID = cms.EDProducer( "EgammaHLTR9IDProducer",
                                             recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
                                             ecalRechitEB = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
                                             ecalRechitEE = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' )
                                )

HLTEgammaR9IDSequence  = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence +
                                       HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate +
                                       hltL1IsoR9ID + hltL1NonIsoR9ID ) 

# create the jetMET HLT reco path
DoHLTJets = cms.Path(HLTBeginSequence + 
    HLTBeginSequence +
    HLTRecoJetSequenceAK5Corrected +
    HLTRegionalRecoJetSequenceAK5Corrected +
    HLTRecoMETSequence +                 
    HLTDoJet30HTRecoSequence
)
DoHLTJetsU = cms.Path(HLTBeginSequence +
    HLTBeginSequence +
    HLTRecoJetSequenceAK5Uncorrected +
    hltMet +
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
    hltL1NonIsoElectronTrackIsol
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
        OpenHLTBLifetimeL3recoSequenceStartup +
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
    HLTDoLocalPixelSequence +
    HLTPixelTrackingForHITrackTrigger + 
    hltPixelCandsForHITrackTrigger +
    hltPixelTracks +
    hltPixelVertices)

hltPixelVertices.beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
