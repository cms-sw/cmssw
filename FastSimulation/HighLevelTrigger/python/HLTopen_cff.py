import FWCore.ParameterSet.Config as cms
 
# import the whole HLT menu
from FastSimulation.Configuration.HLT_cff import *

# create the muon HLT reco path
DoHltMuon = cms.Path(
    HLTBeginSequence +
    HLTL2muonrecoSequence + 
    HLTL2muonisorecoSequence + 
    HLTL3muonrecoSequence + 
    HLTL3muonisorecoSequence +
    cms.SequencePlaceholder("HLTEndSequence") )

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
from FastSimulation.HighLevelTrigger.OpenHLT_Tau_cff import *
DoHLTTau = cms.Path(HLTBeginSequence + 
    #hltTauPrescaler + 
    #hltTauL1SeedFilter + 
    OpenHLTCaloTausCreatorSequence + 
    hltMet + 
    hltL2TauJets + 
    hltL2TauIsolationProducer + 
    hltL2TauIsolationSelector + 
    HLTDoLocalPixelSequence + 
    HLTRecopixelvertexingSequence + 
    hltAssociatorL25Tau + 
    hltConeIsolationL25Tau + 
    hltIsolatedL25Tau + 
    HLTDoLocalStripSequence + 
    hltL3TauPixelSeeds + 
    hltCkfTrackCandidatesL3Tau + 
    hltCtfWithMaterialTracksL3Tau + 
    hltAssociatorL3Tau + 
    hltConeIsolationL3Tau + 
    hltIsolatedL3Tau + 
    TauOpenHLT + 
    cms.SequencePlaceholder("HLTEndSequence"))

# create the b-jet HLT paths
from FastSimulation.HighLevelTrigger.OpenHLT_BJet_cff import *
DoHLTBTag = cms.Path(
    HLTBeginSequence +
    HLTBCommonL2recoSequence +
    OpenHLTBLifetimeL25recoSequence +
    OpenHLTBSoftmuonL25recoSequence +
    OpenHLTBLifetimeL3recoSequence +
    OpenHLTBLifetimeL3recoSequenceRelaxed +
    OpenHLTBSoftmuonL3recoSequence +
    cms.SequencePlaceholder("HLTEndSequence") )
