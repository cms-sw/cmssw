import FWCore.ParameterSet.Config as cms

from ..modules.L1EGammaClusterEmuProducer_cfi import *
from ..modules.l1EGammaEEProducer_cfi import *
from ..modules.l1NNTauProducerPuppi_cfi import *
from ..modules.l1pfCandidates_cfi import *
from ..modules.l1PFMetPuppi_cfi import *
from ..modules.l1pfProducerBarrel_cfi import *
from ..modules.l1pfProducerHF_cfi import *
from ..modules.l1pfProducerHGCal_cfi import *
from ..modules.l1pfProducerHGCalNoTK_cfi import *
from ..modules.L1TkElectronsEllipticMatchCrystal_cfi import *
from ..modules.L1TkElectronsEllipticMatchHGC_cfi import *
from ..modules.L1TkMuons_cfi import *
from ..modules.L1TkPhotonsCrystal_cfi import *
from ..modules.L1TkPhotonsHGC_cfi import *
from ..modules.L1TkPrimaryVertex_cfi import *
from ..modules.l1tSlwPFPuppiJets_cfi import *
from ..modules.l1tSlwPFPuppiJetsCorrected_cfi import *
from ..modules.pfClustersFromCombinedCaloHCal_cfi import *
from ..modules.pfClustersFromCombinedCaloHF_cfi import *
from ..modules.pfClustersFromHGC3DClusters_cfi import *
from ..modules.pfClustersFromL1EGClusters_cfi import *
from ..modules.pfTracksFromL1TracksBarrel_cfi import *
from ..modules.pfTracksFromL1TracksHGCal_cfi import *
from ..modules.simCaloStage2Layer1Digis_cfi import *
from ..modules.simCscTriggerPrimitiveDigis_cfi import *
from ..modules.simDtTriggerPrimitiveDigis_cfi import *
from ..modules.simEmtfDigis_cfi import *
from ..modules.simGmtCaloSumDigis_cfi import *
from ..modules.simGmtStage2Digis_cfi import *
from ..modules.simKBmtfDigis_cfi import *
from ..modules.simKBmtfStubs_cfi import *
from ..modules.simMuonGEMPadDigiClusters_cfi import *
from ..modules.simMuonGEMPadDigis_cfi import *
from ..modules.simOmtfDigis_cfi import *
from ..modules.simTwinMuxDigis_cfi import *

l1tReconstructionTask = cms.Task(
    L1EGammaClusterEmuProducer,
    L1TkElectronsEllipticMatchCrystal,
    L1TkElectronsEllipticMatchHGC,
    L1TkMuons,
    L1TkPhotonsCrystal,
    L1TkPhotonsHGC,
    L1TkPrimaryVertex,
    l1EGammaEEProducer,
    l1NNTauProducerPuppi,
    l1PFMetPuppi,
    l1pfCandidates,
    l1pfProducerBarrel,
    l1pfProducerHF,
    l1pfProducerHGCal,
    l1pfProducerHGCalNoTK,
    l1tSlwPFPuppiJets,
    l1tSlwPFPuppiJetsCorrected,
    pfClustersFromCombinedCaloHCal,
    pfClustersFromCombinedCaloHF,
    pfClustersFromHGC3DClusters,
    pfClustersFromL1EGClusters,
    pfTracksFromL1TracksBarrel,
    pfTracksFromL1TracksHGCal,
    simCaloStage2Layer1Digis,
    simCscTriggerPrimitiveDigis,
    simDtTriggerPrimitiveDigis,
    simEmtfDigis,
    simGmtCaloSumDigis,
    simGmtStage2Digis,
    simKBmtfDigis,
    simKBmtfStubs,
    simMuonGEMPadDigiClusters,
    simMuonGEMPadDigis,
    simOmtfDigis,
    simTwinMuxDigis
)
