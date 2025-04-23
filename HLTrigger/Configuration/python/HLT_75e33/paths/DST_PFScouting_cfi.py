import FWCore.ParameterSet.Config as cms

from ..modules.hltAK4PFJetsForTaus_cfi import *
from ..modules.hltL1GTAcceptFilter_cfi import *
from ..modules.hltFixedGridRhoFastjetAllCaloForEGamma_cfi import *
from ..modules.hltL1SeedsForPuppiMETFilter_cfi import *
from ..modules.hltPFPuppiHT_cfi import *
from ..modules.hltPFPuppiMETTypeOne140_cfi import *
from ..modules.hltPFPuppiMETTypeOneCorrector_cfi import *
from ..modules.hltPFPuppiMETTypeOne_cfi import *
from ..modules.hltPFPuppiMHT140_cfi import *
from ..modules.hltPFPuppiMHT_cfi import *
from ..modules.hltParticleFlowClusterECALUncorrectedUnseeded_cfi import *
from ..modules.hltParticleFlowClusterECALUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitECALUnseeded_cfi import *
from ..modules.hltPhase2L3MuonCandidates_cfi import *
from ..modules.hltEgammaEleL1TrkIsoUnseeded_cfi import *
from ..sequences.HLTAK4PFJetsReconstruction_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTBtagDeepCSVSequencePFPuppi_cfi import *
from ..sequences.HLTBtagDeepFlavourSequencePFPuppi_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTHPSDeepTauPFTauSequence_cfi import *
from ..sequences.HLTHgcalLocalRecoSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTPFPuppiMETReconstruction_cfi import *
from ..sequences.HLTPFTauHPS_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTElePixelMatchUnseededSequence_cfi import *
from ..sequences.HLTGsfElectronUnseededSequence_cfi import *
from ..sequences.HLTPhase2L3MuonGeneralTracksSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaUnseededSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

DST_PFScouting = cms.Path(
    HLTBeginSequence
    + hltL1GTAcceptFilter
    + HLTRawToDigiSequence
    + HLTHgcalLocalRecoSequence
    + HLTLocalrecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + hltParticleFlowRecHitECALUnseeded
    + hltParticleFlowClusterECALUncorrectedUnseeded
    + hltParticleFlowClusterECALUnseeded
    + HLTHgcalTiclPFClusteringForEgammaUnseededSequence
    + HLTPFClusteringForEgammaUnseededSequence
    + hltFixedGridRhoFastjetAllCaloForEGamma
    + HLTElePixelMatchUnseededSequence
    + HLTGsfElectronUnseededSequence
    + hltEgammaEleL1TrkIsoUnseeded
    + hltPhase2L3MuonCandidates
    + HLTPhase2L3MuonGeneralTracksSequence
    + HLTAK4PFJetsReconstruction
    + hltAK4PFJetsForTaus
    + HLTPFTauHPS
    + HLTHPSDeepTauPFTauSequence
    + HLTAK4PFPuppiJetsReconstruction
    + hltPFPuppiHT
    + hltPFPuppiMHT
    + HLTBtagDeepCSVSequencePFPuppi
    + HLTBtagDeepFlavourSequencePFPuppi
    + HLTEndSequence
)
