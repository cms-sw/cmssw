import FWCore.ParameterSet.Config as cms

from ..modules.hlt1PFPuppiCentralJet75MaxEta2p4_cfi import *
from ..modules.hlt2PFPuppiCentralJet60MaxEta2p4_cfi import *
from ..modules.hlt3PFPuppiCentralJet45MaxEta2p4_cfi import *
from ..modules.hlt4PFPuppiCentralJet40MaxEta2p4_cfi import *
from ..modules.hltBTagPFPuppiDeepFlavour0p275Eta2p4TripleEta2p4_cfi import *
from ..modules.hltHtMhtPFPuppiCentralJetsQuadC30MaxEta2p4_cfi import *
from ..modules.hltPFPuppiCentralJetQuad30MaxEta2p4_cfi import *
from ..modules.hltPFPuppiCentralJetsQuad30HT330MaxEta2p4_cfi import *
from ..modules.hltL1SeedsForQuadPuppiJetTripleBtagFilter_cfi import *
from ..sequences.hgcalLocalRecoSequence_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTBtagDeepFlavourSequencePFPuppiModEta2p4_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *
from ..sequences.localrecoSequence_cfi import *
from ..sequences.RawToDigiSequence_cfi import *

HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4 = cms.Path(HLTBeginSequence+hltL1SeedsForQuadPuppiJetTripleBtagFilter+RawToDigiSequence+hgcalLocalRecoSequence+localrecoSequence+HLTTrackingV61Sequence+HLTMuonsSequence+HLTParticleFlowSequence+HLTAK4PFPuppiJetsReconstruction+hltPFPuppiCentralJetQuad30MaxEta2p4+hlt1PFPuppiCentralJet75MaxEta2p4+hlt2PFPuppiCentralJet60MaxEta2p4+hlt3PFPuppiCentralJet45MaxEta2p4+hlt4PFPuppiCentralJet40MaxEta2p4+hltHtMhtPFPuppiCentralJetsQuadC30MaxEta2p4+hltPFPuppiCentralJetsQuad30HT330MaxEta2p4+HLTBtagDeepFlavourSequencePFPuppiModEta2p4+hltBTagPFPuppiDeepFlavour0p275Eta2p4TripleEta2p4+HLTEndSequence)
