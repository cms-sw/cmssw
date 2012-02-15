import FWCore.ParameterSet.Config as cms
 
# import the whole HLT menu
from HLTrigger.HLTanalyzers.HLT_FULL_cff import *

########################################################
# Customizations
########################################################

# HBHE noise
from HLTrigger.HLTanalyzers.OpenHLT_HBHEnoise_cff import *

# BTag
from HLTrigger.HLTanalyzers.OpenHLT_BJet_cff import *

# Tau
from HLTrigger.HLTanalyzers.OpenHLT_Tau_cff import *

# Dimuon TP
hltMuTrackJpsiPixelTrackSelector.MinMasses  = cms.vdouble ( 2.0, 60.0 )
hltMuTrackJpsiPixelTrackSelector.MaxMasses  = cms.vdouble ( 4.6, 120.0 )
hltMu5Track1JpsiPixelMassFiltered.MinMasses = cms.vdouble ( 2.0, 60.0 )
hltMu5Track1JpsiPixelMassFiltered.MaxMasses = cms.vdouble ( 4.6, 120.0 )
hltMu5TkMuJpsiTrackMassFiltered.MinMasses   = cms.vdouble ( 2.5, 60.0 )
hltMu5TkMuJpsiTrackMassFiltered.MaxMasses   = cms.vdouble ( 4.1, 120.0 )
hltMu5Track2JpsiTrackMassFiltered.MinMasses = cms.vdouble ( 2.7, 60.0 )
hltMu5Track2JpsiTrackMassFiltered.MaxMasses = cms.vdouble ( 3.5, 120.0 )

hltMu5L2Mu2JpsiTrackMassFiltered.MinMasses = cms.vdouble ( 1.8, 50.0 )
hltMu5L2Mu2JpsiTrackMassFiltered.MaxMasses = cms.vdouble ( 4.5, 130.0 )

########################################################
# Paths without filters
########################################################

# create the jetMET HLT reco path
DoHLTJets = cms.Path(
    HLTBeginSequence +
    HLTRecoJetSequenceAK5Corrected +
    HLTRecoJetSequenceAK5L1FastJetCorrected +
    HLTRecoMETSequence +
    HLTDoLocalHcalWithoutHOSequence +                 
    OpenHLTHCalNoiseTowerCleanerSequence
)
DoHLTJetsU = cms.Path(HLTBeginSequence +
    HLTBeginSequence +
    HLTRecoJetSequenceAK5Uncorrected +
    HLTRecoMETSequence
)

# create the muon HLT reco path
DoHltMuon = cms.Path(
    HLTBeginSequence +
    HLTL2muonrecoSequenceNoVtx +
    HLTL2muonrecoSequence + 
    HLTL2muonisorecoSequence + 
    HLTL3muonrecoSequence + 
    HLTL3muonisorecoSequence +
    HLTL3muonTkIso10recoSequence + 
    HLTMuTrackJpsiPixelRecoSequence + 
    HLTMuTrackJpsiTrackRecoSequence +
##    HLTDisplacemumuSequence +

    HLTDoLocalPixelSequence +
    hltPixelTracks +
    HLTDoLocalStripSequence +
    hltMuTrackSeeds +
    hltMuCkfTrackCandidates +
    hltMuCtfTracks +
    hltDiMuonMerging +
    HLTL3muonrecoNocandSequence +
    hltDiMuonLinks +
    hltGlbTrkMuons +
    hltGlbTrkMuonCands +
    
    HLTEndSequence )

# create the Egamma HLT reco paths
DoHLTPhoton = cms.Path( 
    HLTBeginSequence + 
    HLTDoRegionalEgammaEcalSequence + 
    HLTL1IsolatedEcalClustersSequence + 
    HLTL1NonIsolatedEcalClustersSequence + 
    hltL1IsoRecoEcalCandidate + 
    hltL1NonIsoRecoEcalCandidate + 
    HLTEgammaR9IDSequence +
    hltL1IsolatedPhotonEcalIsol + 
    hltL1NonIsolatedPhotonEcalIsol + 
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
    hltL1NonIsolatedPhotonHollowTrackIsol +
    HLTEcalActivitySequence +
    hltActivityPhotonHcalForHE +
    hltActivityR9ID +
    hltActivityPhotonClusterShape +
    hltActivityPhotonEcalIsol +
    hltActivityPhotonHcalIsol +
    HLTEcalActivityEgammaRegionalRecoTrackerSequence +
    hltEcalActivityEgammaRegionalAnalyticalTrackSelector + 
    hltActivityPhotonHollowTrackIsolWithId
    ##    hltActivityPhotonHollowTrackIsol
    )

DoHLTElectron = cms.Path(
    HLTBeginSequence +
    HLTDoRegionalEgammaEcalSequence +
    HLTL1IsolatedEcalClustersSequence +
    HLTL1NonIsolatedEcalClustersSequence +
    hltL1IsoRecoEcalCandidate +
    hltL1NonIsoRecoEcalCandidate +
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
    hltHFRecoEcalTightCandidate
)


# create the tau HLT reco path
DoHLTTau = cms.Path(HLTBeginSequence +
                    OpenHLTCaloTausCreatorSequence +
                    openhltL2TauJets +
                    openhltL2TauIsolationProducer +
                    HLTDoLocalPixelSequence +
                    HLTRecopixelvertexingSequence +
                    OpenHLTL25TauTrackReconstructionSequence +
                    OpenHLTL25TauTrackIsolation +
                    TauOpenHLT+
                    HLTRecoJetSequencePrePF +
                    HLTPFJetTriggerSequence +
                    pfAllMuons +
                    HLTPFJetTriggerSequenceForTaus +
                    HLTPFTauSequence +
                    HLTEndSequence)

# create the b-jet HLT paths
DoHLTBTag = cms.Path(
        HLTBeginSequence +
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
    HLTDoLocalPixelSequence +
    HLTDoHILocalPixelSequence +
    HLTPixelTrackingForHITrackTrigger + 
    hltPixelCandsForHITrackTrigger +
    hltPixelTracks +
    hltPixelVertices)

