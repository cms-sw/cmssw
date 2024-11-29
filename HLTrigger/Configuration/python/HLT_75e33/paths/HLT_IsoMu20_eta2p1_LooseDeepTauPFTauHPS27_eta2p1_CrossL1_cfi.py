import FWCore.ParameterSet.Config as cms

from ..sequences.HLTHgcalLocalRecoSequence_cfi import *
from ..sequences.HLTAK4PFJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTPFTauHPS_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTHPSDeepTauPFTauSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTPhase2L3MuonGeneralTracksSequence_cfi import *
from ..modules.hltAK4PFJetsForTaus_cfi import *
from ..modules.hltPhase2L3MuonCandidates_cfi import *
from ..modules.hltHpsSelectedPFTauLooseTauWPDeepTau_cfi import *
from ..modules.hltHpsPFTau27LooseTauWPDeepTau_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22EcalIso0p41_cfi import *
from ..modules.hltPhase2L3MuonsHgcalLCIsodR0p2dRVetoEM0p00dRVetoHad0p02minEEM0p00minEHad0p00_cfi import *
from ..modules.hltPhase2L3MuonsHcalIsodR0p3dRVeto0p000_cfi import *
from ..modules.hltPhase2L3MuonsEcalIsodR0p3dRVeto0p000_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22HcalIso0p40_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22HgcalIso4p70_cfi import *
from ..modules.hltFixedGridRhoFastjetAllCaloForEGamma_cfi import *
from ..modules.hltParticleFlowClusterECALUnseeded_cfi import *
from ..modules.hltParticleFlowClusterECALUncorrectedUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitECALUnseeded_cfi import *
from ..modules.hltPuppiTauTkMuon4218L1TkFilter_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22TrkIsoRegionalNewFiltered0p07EcalHcalHgcalTrk_cfi import *
from ..modules.hltL3fL1TkSingleMu18Filtered20_cfi import *
from ..modules.hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p07_cfi import *

HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1 = cms.Path(
    HLTBeginSequence
    + hltPuppiTauTkMuon4218L1TkFilter
    + HLTRawToDigiSequence
    + HLTHgcalLocalRecoSequence
    + HLTLocalrecoSequence
    + HLTTrackingV61Sequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + hltParticleFlowRecHitECALUnseeded
    + hltParticleFlowClusterECALUncorrectedUnseeded
    + hltParticleFlowClusterECALUnseeded
    + hltFixedGridRhoFastjetAllCaloForEGamma
    + hltPhase2L3MuonCandidates
    + hltPhase2L3MuonsEcalIsodR0p3dRVeto0p000
    + hltPhase2L3MuonsHcalIsodR0p3dRVeto0p000
    + hltPhase2L3MuonsHgcalLCIsodR0p2dRVetoEM0p00dRVetoHad0p02minEEM0p00minEHad0p00
    + hltL3fL1TkSingleMu18Filtered20
    + hltL3crIsoL1TkSingleMu22EcalIso0p41
    + hltL3crIsoL1TkSingleMu22HcalIso0p40
    + hltL3crIsoL1TkSingleMu22HgcalIso4p70
    + HLTPhase2L3MuonGeneralTracksSequence
    + hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p07
    + hltL3crIsoL1TkSingleMu22TrkIsoRegionalNewFiltered0p07EcalHcalHgcalTrk
    + HLTAK4PFJetsReconstruction
    + hltAK4PFJetsForTaus
    + HLTPFTauHPS
    + HLTHPSDeepTauPFTauSequence
    + hltHpsSelectedPFTauLooseTauWPDeepTau
    + hltHpsPFTau27LooseTauWPDeepTau
    + HLTEndSequence
)
