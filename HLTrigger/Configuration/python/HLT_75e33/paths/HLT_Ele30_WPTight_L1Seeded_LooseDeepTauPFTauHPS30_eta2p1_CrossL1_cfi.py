import FWCore.ParameterSet.Config as cms
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *
from ..sequences.HLTElePixelMatchL1SeededSequence_cfi import *
from ..sequences.HLTGsfElectronL1SeededSequence_cfi import *
from ..sequences.HLTAK4PFJetsReconstruction_cfi import *
from ..sequences.HLTPFTauHPS_cfi import *
from ..sequences.HLTHPSDeepTauPFTauSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *
from ..sequences.HLTHgcalLocalRecoSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..modules.hltPuppiTauTkIsoEle45_22L1TkFilter_cfi import *
from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEG30EtL1SeededFilter_cfi import * 
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoV6L1Seeded_cfi import *
from ..modules.hltAK4PFJetsForTaus_cfi import *
from ..modules.hltHpsSelectedPFTauLooseTauWPDeepTau_cfi import * 
from ..modules.hltHpsPFTau30LooseTauWPDeepTau_cfi import *
from ..modules.hltEle30WPTightClusterShapeL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightClusterShapeSigmavvL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightClusterShapeSigmawwL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightHgcalHEL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightHEL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightEcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightHgcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightHcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightPixelMatchL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightPMS2L1SeededFilter_cfi import *
from ..modules.hltEle30WPTightGsfOneOEMinusOneOPL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightGsfDetaL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightGsfDphiL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightBestGsfNLayerITL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightBestGsfChi2L1SeededFilter_cfi import *
from ..modules.hltEle30WPTightGsfTrackIsoFromL1TracksL1SeededFilter_cfi import *
from ..modules.hltEle30WPTightGsfTrackIsoL1SeededFilter_cfi import *


HLT_Ele30_WPTight_L1Seeded_LooseDeepTauPFTauHPS30_eta2p1_CrossL1 = cms.Path( 
    HLTBeginSequence +
    hltPuppiTauTkIsoEle45_22L1TkFilter +
    HLTRawToDigiSequence +
    HLTHgcalLocalRecoSequence +
    HLTLocalrecoSequence +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1SeededSequence +
    HLTHgcalTiclPFClusteringForEgammaL1SeededSequence +
    hltEgammaCandidatesL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltEG30EtL1SeededFilter + 
    hltEgammaClusterShapeL1Seeded + 
    hltEle30WPTightClusterShapeL1SeededFilter + 
    hltEgammaHGCALIDVarsL1Seeded +
    hltEle30WPTightClusterShapeSigmavvL1SeededFilter + 
    hltEle30WPTightClusterShapeSigmawwL1SeededFilter + 
    hltEle30WPTightHgcalHEL1SeededFilter + 
    HLTEGammaDoLocalHcalSequence + 
    HLTFastJetForEgammaSequence +
    hltEgammaHoverEL1Seeded +
    hltEle30WPTightHEL1SeededFilter + 
    hltEgammaEcalPFClusterIsoL1Seeded +
    hltEle30WPTightEcalIsoL1SeededFilter + 
    hltEgammaHGCalLayerClusterIsoL1Seeded + 
    hltEle30WPTightHgcalIsoL1SeededFilter + 
    HLTPFHcalClusteringForEgammaSequence +
    hltEgammaHcalPFClusterIsoL1Seeded + 
    hltEle30WPTightHcalIsoL1SeededFilter + 
    HLTElePixelMatchL1SeededSequence + 
    hltEle30WPTightPixelMatchL1SeededFilter + 
    hltEle30WPTightPMS2L1SeededFilter + 
    HLTGsfElectronL1SeededSequence + 
    hltEle30WPTightGsfOneOEMinusOneOPL1SeededFilter + 
    hltEle30WPTightGsfDetaL1SeededFilter + 
    hltEle30WPTightGsfDphiL1SeededFilter + 
    hltEle30WPTightBestGsfNLayerITL1SeededFilter + 
    hltEle30WPTightBestGsfChi2L1SeededFilter + 
    hltEgammaEleL1TrkIsoL1Seeded + 
    hltEle30WPTightGsfTrackIsoFromL1TracksL1SeededFilter + 
    HLTTrackingV61Sequence +
    hltEgammaEleGsfTrackIsoV6L1Seeded +
    hltEle30WPTightGsfTrackIsoL1SeededFilter + 
    HLTMuonsSequence +
    HLTParticleFlowSequence +
    HLTAK4PFJetsReconstruction +
    hltAK4PFJetsForTaus +
    HLTPFTauHPS +
    HLTHPSDeepTauPFTauSequence +
    hltHpsSelectedPFTauLooseTauWPDeepTau +
    hltHpsPFTau30LooseTauWPDeepTau +
    HLTEndSequence
)
