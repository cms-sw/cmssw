import FWCore.ParameterSet.Config as cms

from ..modules.hlt1PFPuppiCentralJet70MaxEta2p4_cfi import *
from ..modules.hlt2PFPuppiCentralJet40MaxEta2p4_cfi import *
from ..modules.hltBTagPFPuppiDeepFlavour0p375Eta2p4TripleEta2p4_cfi import *
from ..modules.hltHtMhtPFPuppiCentralJetsQuadC30MaxEta2p4_cfi import *
from ..modules.hltPFPuppiCentralJetQuad30MaxEta2p4_cfi import *
from ..modules.hltPFPuppiCentralJetsQuad30HT200MaxEta2p4_cfi import *
from ..modules.hltL1SeedsForQuadPuppiJetTripleBtagFilter_cfi import *
from ..sequences.HLTHgcalLocalRecoSequence_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTBtagDeepFlavourSequencePFPuppiModEta2p4_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *

HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4 = cms.Path(
    HLTBeginSequence
    + hltL1SeedsForQuadPuppiJetTripleBtagFilter
    + HLTRawToDigiSequence
    + HLTHgcalLocalRecoSequence
    + HLTLocalrecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTAK4PFPuppiJetsReconstruction
    + hltPFPuppiCentralJetQuad30MaxEta2p4
    + hlt1PFPuppiCentralJet70MaxEta2p4
    + hlt2PFPuppiCentralJet40MaxEta2p4
    + hltHtMhtPFPuppiCentralJetsQuadC30MaxEta2p4
    + hltPFPuppiCentralJetsQuad30HT200MaxEta2p4
    + HLTBtagDeepFlavourSequencePFPuppiModEta2p4
    + hltBTagPFPuppiDeepFlavour0p375Eta2p4TripleEta2p4
    + HLTEndSequence
)
