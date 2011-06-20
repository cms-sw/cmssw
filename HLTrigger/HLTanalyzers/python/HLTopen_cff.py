import FWCore.ParameterSet.Config as cms
 
# import the whole HLT menu
from HLTrigger.HLTanalyzers.HLT_FULL_cff import *

hltL1IsoR9shape = cms.EDProducer( "EgammaHLTR9Producer",
                                  recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
                                  ecalRechitEB = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
                                  ecalRechitEE = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
                                 useSwissCross = cms.bool( False )
                                  )
hltL1NonIsoR9shape = cms.EDProducer( "EgammaHLTR9Producer",
                                     recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
                                     ecalRechitEB = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
                                     ecalRechitEE = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
                                     useSwissCross = cms.bool( False )
                                     )

HLTEgammaR9ShapeSequence = cms.Sequence( hltL1IsoR9shape + hltL1NonIsoR9shape )

hltLowMassDisplacedL3Filtered.MaxEta      = cms.double(3.0)
hltLowMassDisplacedL3Filtered.MinPtPair   = cms.double( 0.0 )
hltLowMassDisplacedL3Filtered.MinPtMin    = cms.double( 0.0 )
hltLowMassDisplacedL3Filtered.MaxInvMass  = cms.double( 11.5 )

hltDisplacedmumuFilterLowMass.MinLxySignificance     = cms.double( 0.0 )
hltDisplacedmumuFilterLowMass.MinVtxProbability      = cms.double( 0.0 )
hltDisplacedmumuFilterLowMass.MinCosinePointingAngle = cms.double( -2.0 )


HLTDisplacemumuSequence = cms.Sequence(  hltL1sL1DoubleMu0 + hltDimuonL1Filtered0 + hltDimuonL2PreFiltered0 + hltLowMassDisplacedL3Filtered + hltDisplacedmumuVtxProducerLowMass + hltDisplacedmumuFilterLowMass)



# create the jetMET HLT reco path
DoHLTJets = cms.Path(HLTBeginSequence + 
    HLTBeginSequence +
    HLTRecoJetSequenceAK5Corrected +
    HLTRecoMETSequence +
    HLTDoLocalHcalWithoutHOSequence                  
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
    HLTMuTrackJpsiPixelRecoSequence + 
    HLTMuTrackJpsiTrackRecoSequence +
    HLTDisplacemumuSequence +
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
    hltActivityPhotonHollowTrackIsol
    )

DoHLTElectron = cms.Path(
    HLTBeginSequence +
    HLTDoRegionalEgammaEcalSequence +
    HLTL1IsolatedEcalClustersSequence +
    HLTL1NonIsolatedEcalClustersSequence +
    hltL1IsoRecoEcalCandidate +
    hltL1NonIsoRecoEcalCandidate +
    HLTEgammaR9ShapeSequence +#was commented out for HT jobs
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
                    HLTDoLocalPixelSequence +
                    HLTRecopixelvertexingSequence +
                    OpenHLTL25TauTrackReconstructionSequence +
                    OpenHLTL25TauTrackIsolation +
                    TauOpenHLT+
                    HLTRecoJetSequencePrePF +
                    HLTPFJetTriggerSequence +
                    HLTPFJetTriggerSequenceForTaus +
                    HLTPFTauSequence +
                    HLTEndSequence)

# create the b-jet HLT paths
from HLTrigger.HLTanalyzers.OpenHLT_BJet_cff import *
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
    # HLTDoLocalPixelSequence +
    HLTDoHILocalPixelSequence +
    HLTPixelTrackingForHITrackTrigger + 
    hltPixelCandsForHITrackTrigger +
    hltPixelTracks +
    hltPixelVertices)

