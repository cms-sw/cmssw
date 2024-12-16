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
from ..modules.hltEle32WPTightClusterShapeL1SeededFilter_cfi import hltEle32WPTightClusterShapeL1SeededFilter as _hltEle32WPTightClusterShapeL1SeededFilter
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEle32WPTightClusterShapeSigmavvL1SeededFilter_cfi import hltEle32WPTightClusterShapeSigmavvL1SeededFilter as _hltEle32WPTightClusterShapeSigmavvL1SeededFilter
from ..modules.hltEle32WPTightClusterShapeSigmawwL1SeededFilter_cfi import hltEle32WPTightClusterShapeSigmawwL1SeededFilter as _hltEle32WPTightClusterShapeSigmawwL1SeededFilter
from ..modules.hltEle32WPTightHgcalHEL1SeededFilter_cfi import hltEle32WPTightHgcalHEL1SeededFilter as _hltEle32WPTightHgcalHEL1SeededFilter
from ..modules.hltEgammaHoverEL1Seeded_cfi import *
from ..modules.hltEle32WPTightHEL1SeededFilter_cfi import hltEle32WPTightHEL1SeededFilter as _hltEle32WPTightHEL1SeededFilter
from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEle32WPTightEcalIsoL1SeededFilter_cfi import hltEle32WPTightEcalIsoL1SeededFilter as _hltEle32WPTightEcalIsoL1SeededFilter
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *
from ..modules.hltEle32WPTightHgcalIsoL1SeededFilter_cfi import hltEle32WPTightHgcalIsoL1SeededFilter as _hltEle32WPTightHgcalIsoL1SeededFilter
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEle32WPTightHcalIsoL1SeededFilter_cfi import hltEle32WPTightHcalIsoL1SeededFilter as _hltEle32WPTightHcalIsoL1SeededFilter
from ..modules.hltEle32WPTightPixelMatchL1SeededFilter_cfi import hltEle32WPTightPixelMatchL1SeededFilter as _hltEle32WPTightPixelMatchL1SeededFilter
from ..modules.hltEle32WPTightPMS2L1SeededFilter_cfi import hltEle32WPTightPMS2L1SeededFilter as _hltEle32WPTightPMS2L1SeededFilter
from ..modules.hltEle32WPTightGsfOneOEMinusOneOPL1SeededFilter_cfi import hltEle32WPTightGsfOneOEMinusOneOPL1SeededFilter as _hltEle32WPTightGsfOneOEMinusOneOPL1SeededFilter
from ..modules.hltEle32WPTightGsfDetaL1SeededFilter_cfi import hltEle32WPTightGsfDetaL1SeededFilter as _hltEle32WPTightGsfDetaL1SeededFilter
from ..modules.hltEle32WPTightGsfDphiL1SeededFilter_cfi import hltEle32WPTightGsfDphiL1SeededFilter as _hltEle32WPTightGsfDphiL1SeededFilter
from ..modules.hltEle32WPTightBestGsfNLayerITL1SeededFilter_cfi import hltEle32WPTightBestGsfNLayerITL1SeededFilter as _hltEle32WPTightBestGsfNLayerITL1SeededFilter
from ..modules.hltEle32WPTightBestGsfChi2L1SeededFilter_cfi import hltEle32WPTightBestGsfChi2L1SeededFilter as _hltEle32WPTightBestGsfChi2L1SeededFilter
from ..modules.hltEgammaEleL1TrkIsoL1Seeded_cfi import *
from ..modules.hltEle32WPTightGsfTrackIsoFromL1TracksL1SeededFilter_cfi import hltEle32WPTightGsfTrackIsoFromL1TracksL1SeededFilter as _hltEle32WPTightGsfTrackIsoFromL1TracksL1SeededFilter
from ..modules.hltEgammaEleGsfTrackIsoV6L1Seeded_cfi import *
from ..modules.hltEle32WPTightGsfTrackIsoL1SeededFilter_cfi import hltEle32WPTightGsfTrackIsoL1SeededFilter as _hltEle32WPTightGsfTrackIsoL1SeededFilter
from ..modules.hltAK4PFJetsForTaus_cfi import *
from ..modules.hltHpsSelectedPFTauLooseTauWPDeepTau_cfi import * 
from ..modules.hltHpsPFTau30LooseTauWPDeepTau_cfi import *


hltEle30WPTightClusterShapeL1SeededFilter = _hltEle32WPTightClusterShapeL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag('hltEG30EtL1SeededFilter'),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(0),
    thrOverE2EE = cms.vdouble(0),
    thrOverEEB = cms.vdouble(0),
    thrOverEEE = cms.vdouble(0),
    thrRegularEB = cms.vdouble(0.013),
    thrRegularEE = cms.vdouble(0.013),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaClusterShapeL1Seeded","sigmaIEtaIEta5x5")
)

hltEle30WPTightClusterShapeSigmavvL1SeededFilter = _hltEle32WPTightClusterShapeSigmavvL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag("hltEle30WPTightClusterShapeL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(0),
    thrOverE2EE = cms.vdouble(0),
    thrOverEEB = cms.vdouble(0.0008),
    thrOverEEE = cms.vdouble(0.0008),
    thrRegularEB = cms.vdouble(0.7225),
    thrRegularEE = cms.vdouble(0.7225),
    useEt = cms.bool(True),
    varTag = cms.InputTag("hltEgammaHGCALIDVarsL1Seeded","sigma2vv")    
)

hltEle30WPTightClusterShapeSigmawwL1SeededFilter = _hltEle32WPTightClusterShapeSigmawwL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag("hltEle30WPTightClusterShapeSigmavvL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(0),
    thrOverE2EE = cms.vdouble(0),
    thrOverEEB = cms.vdouble(0.04),
    thrOverEEE = cms.vdouble(0.04),
    thrRegularEB = cms.vdouble(72.25),
    thrRegularEE = cms.vdouble(72.25),
    useEt = cms.bool(True),
    varTag = cms.InputTag("hltEgammaHGCALIDVarsL1Seeded","sigma2ww")
)


hltEle30WPTightHgcalHEL1SeededFilter = _hltEle32WPTightHgcalHEL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.0, 1.479, 2.1),
    candTag = cms.InputTag("hltEle30WPTightClusterShapeSigmawwL1SeededFilter"), 
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0, 0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    etaBoundaryEB12 = cms.double(1.0),
    etaBoundaryEE12 = cms.double(2.1),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag("hltFixedGridRhoFastjetAllCaloForEGamma"),
    saveTags = cms.bool(True),
    thrOverE2EB1 = cms.vdouble(0.0),
    thrOverE2EB2 = cms.vdouble(0.0),
    thrOverE2EE1 = cms.vdouble(0.0),
    thrOverE2EE2 = cms.vdouble(0.0),
    thrOverEEB1 = cms.vdouble(0.0),
    thrOverEEB2 = cms.vdouble(0.0),
    thrOverEEE1 = cms.vdouble(0.15),
    thrOverEEE2 = cms.vdouble(0.15),
    thrRegularEB1 = cms.vdouble(9999.0),
    thrRegularEB2 = cms.vdouble(9999.0),
    thrRegularEE1 = cms.vdouble(5.0),
    thrRegularEE2 = cms.vdouble(5.0),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaHGCALIDVarsL1Seeded","hForHOverE")
)

hltEle30WPTightHEL1SeededFilter = _hltEle32WPTightHEL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.0, 1.479, 2.1),
    candTag = cms.InputTag("hltEle30WPTightHgcalHEL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.1, 0.1, 0.3, 0.5),
    energyLowEdges = cms.vdouble(0.0),
    etaBoundaryEB12 = cms.double(1.0),
    etaBoundaryEE12 = cms.double(2.1),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag("hltFixedGridRhoFastjetAllCaloForEGamma"),
    saveTags = cms.bool(True),
    thrOverE2EB1 = cms.vdouble(0.0),
    thrOverE2EB2 = cms.vdouble(0.0),
    thrOverE2EE1 = cms.vdouble(0.0),
    thrOverE2EE2 = cms.vdouble(0.0),
    thrOverEEB1 = cms.vdouble(0.175),
    thrOverEEB2 = cms.vdouble(0.175),
    thrOverEEE1 = cms.vdouble(0.0),
    thrOverEEE2 = cms.vdouble(0.0),
    thrRegularEB1 = cms.vdouble(0.0),
    thrRegularEB2 = cms.vdouble(0.0),
    thrRegularEE1 = cms.vdouble(9999.0),
    thrRegularEE2 = cms.vdouble(9999.0),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaHoverEL1Seeded")
)

hltEle30WPTightEcalIsoL1SeededFilter = _hltEle32WPTightEcalIsoL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.0, 1.479, 2.1),
    candTag = cms.InputTag("hltEle30WPTightHEL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.2, 0.2, 0.25, 0.3),
    energyLowEdges = cms.vdouble(0.0),
    etaBoundaryEB12 = cms.double(1.0),
    etaBoundaryEE12 = cms.double(2.1),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag("hltFixedGridRhoFastjetAllCaloForEGamma"),
    saveTags = cms.bool(True),
    thrOverE2EB1 = cms.vdouble(0.0),
    thrOverE2EB2 = cms.vdouble(0.0),
    thrOverE2EE1 = cms.vdouble(0.0),
    thrOverE2EE2 = cms.vdouble(0.0),
    thrOverEEB1 = cms.vdouble(0.02),
    thrOverEEB2 = cms.vdouble(0.02),
    thrOverEEE1 = cms.vdouble(0.02),
    thrOverEEE2 = cms.vdouble(0.02),
    thrRegularEB1 = cms.vdouble(9.0),
    thrRegularEB2 = cms.vdouble(9.0),
    thrRegularEE1 = cms.vdouble(9.0),
    thrRegularEE2 = cms.vdouble(9.0),
    useEt = cms.bool(True),
    varTag = cms.InputTag("hltEgammaEcalPFClusterIsoL1Seeded")
)

hltEle30WPTightHgcalIsoL1SeededFilter = _hltEle32WPTightHgcalIsoL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.0, 1.479, 2.0),
    candTag = cms.InputTag("hltEle30WPTightEcalIsoL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0, 0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    etaBoundaryEB12 = cms.double(1.0),
    etaBoundaryEE12 = cms.double(2.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB1 = cms.vdouble(0.0),
    thrOverE2EB2 = cms.vdouble(0.0),
    thrOverE2EE1 = cms.vdouble(0.0),
    thrOverE2EE2 = cms.vdouble(0.0),
    thrOverEEB1 = cms.vdouble(0.05),
    thrOverEEB2 = cms.vdouble(0.05),
    thrOverEEE1 = cms.vdouble(0.05),
    thrOverEEE2 = cms.vdouble(0.05),
    thrRegularEB1 = cms.vdouble(150),
    thrRegularEB2 = cms.vdouble(150),
    thrRegularEE1 = cms.vdouble(150),
    thrRegularEE2 = cms.vdouble(350),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaHGCalLayerClusterIsoL1Seeded")
)

hltEle30WPTightHcalIsoL1SeededFilter = _hltEle32WPTightHcalIsoL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.0, 1.479, 2.0),
    candTag = cms.InputTag("hltEle30WPTightHgcalIsoL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.2, 0.2, 0.4, 0.5),
    energyLowEdges = cms.vdouble(0.0),
    etaBoundaryEB12 = cms.double(1.0),
    etaBoundaryEE12 = cms.double(2.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag("hltFixedGridRhoFastjetAllCaloForEGamma"),
    saveTags = cms.bool(True),
    thrOverE2EB1 = cms.vdouble(0.0),
    thrOverE2EB2 = cms.vdouble(0.0),
    thrOverE2EE1 = cms.vdouble(0.0),
    thrOverE2EE2 = cms.vdouble(0.0),
    thrOverEEB1 = cms.vdouble(0.02),
    thrOverEEB2 = cms.vdouble(0.02),
    thrOverEEE1 = cms.vdouble(0.02),
    thrOverEEE2 = cms.vdouble(0.02),
    thrRegularEB1 = cms.vdouble(19),
    thrRegularEB2 = cms.vdouble(19),
    thrRegularEE1 = cms.vdouble(19),
    thrRegularEE2 = cms.vdouble(19),
    useEt = cms.bool(True),
    varTag = cms.InputTag("hltEgammaHcalPFClusterIsoL1Seeded")
)


hltEle30WPTightPixelMatchL1SeededFilter = _hltEle32WPTightPixelMatchL1SeededFilter.clone(
    candTag = cms.InputTag("hltEle30WPTightHcalIsoL1SeededFilter"),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    l1PixelSeedsTag = cms.InputTag("hltEgammaElectronPixelSeedsL1Seeded"),
    ncandcut = cms.int32(1),
    npixelmatchcut = cms.double(1.0),
    pixelVeto = cms.bool(False),
    s2_threshold = cms.double(0.4),
    s_a_phi1B = cms.double(0.0069),
    s_a_phi1F = cms.double(0.0076),
    s_a_phi1I = cms.double(0.0088),
    s_a_phi2B = cms.double(0.00037),
    s_a_phi2F = cms.double(0.00906),
    s_a_phi2I = cms.double(0.0007),
    s_a_rF = cms.double(0.04),
    s_a_rI = cms.double(0.027),
    s_a_zB = cms.double(0.012),
    saveTags = cms.bool(True),
    tanhSO10BarrelThres = cms.double(0.35),
    tanhSO10ForwardThres = cms.double(1.0),
    tanhSO10InterThres = cms.double(1.0),
    useS = cms.bool(False)
)

hltEle30WPTightPMS2L1SeededFilter = _hltEle32WPTightPMS2L1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag("hltEle30WPTightPixelMatchL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(0),
    thrOverE2EE = cms.vdouble(0),
    thrOverEEB = cms.vdouble(0),
    thrOverEEE = cms.vdouble(0),
    thrRegularEB = cms.vdouble(55.0),
    thrRegularEE = cms.vdouble(75.0),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaPixelMatchVarsL1Seeded","s2")
)


hltEle30WPTightGsfOneOEMinusOneOPL1SeededFilter = _hltEle32WPTightGsfOneOEMinusOneOPL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 0.8, 1.479, 2.1),
    candTag = cms.InputTag("hltEle30WPTightPMS2L1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0, 0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    etaBoundaryEB12 = cms.double(0.8),
    etaBoundaryEE12 = cms.double(2.1),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB1 = cms.vdouble(0.0),
    thrOverE2EB2 = cms.vdouble(0.0),
    thrOverE2EE1 = cms.vdouble(0.0),
    thrOverE2EE2 = cms.vdouble(0.0),
    thrOverEEB1 = cms.vdouble(0.0),
    thrOverEEB2 = cms.vdouble(0.0),
    thrOverEEE1 = cms.vdouble(0.0),
    thrOverEEE2 = cms.vdouble(0.0),
    thrRegularEB1 = cms.vdouble(0.04),
    thrRegularEB2 = cms.vdouble(0.08),
    thrRegularEE1 = cms.vdouble(0.04),
    thrRegularEE2 = cms.vdouble(0.04),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaGsfTrackVarsL1Seeded","OneOESuperMinusOneOP")
)

hltEle30WPTightGsfDetaL1SeededFilter = _hltEle32WPTightGsfDetaL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 0.8, 1.479, 2.1),
    candTag = cms.InputTag("hltEle30WPTightGsfOneOEMinusOneOPL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0, 0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    etaBoundaryEB12 = cms.double(0.8),
    etaBoundaryEE12 = cms.double(2.1),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB1 = cms.vdouble(0.0),
    thrOverE2EB2 = cms.vdouble(0.0),
    thrOverE2EE1 = cms.vdouble(0.0),
    thrOverE2EE2 = cms.vdouble(0.0),
    thrOverEEB1 = cms.vdouble(0.0),
    thrOverEEB2 = cms.vdouble(0.0),
    thrOverEEE1 = cms.vdouble(0.0),
    thrOverEEE2 = cms.vdouble(0.0),
    thrRegularEB1 = cms.vdouble(0.003),
    thrRegularEB2 = cms.vdouble(0.009),
    thrRegularEE1 = cms.vdouble(0.004),
    thrRegularEE2 = cms.vdouble(0.004),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaGsfTrackVarsL1Seeded","DetaSeed")
)

hltEle30WPTightGsfDphiL1SeededFilter = _hltEle32WPTightGsfDphiL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 0.8, 1.479, 2.1),
    candTag = cms.InputTag("hltEle30WPTightGsfDetaL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0, 0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    etaBoundaryEB12 = cms.double(0.8),
    etaBoundaryEE12 = cms.double(2.1),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB1 = cms.vdouble(0.0),
    thrOverE2EB2 = cms.vdouble(0.0),
    thrOverE2EE1 = cms.vdouble(0.0),
    thrOverE2EE2 = cms.vdouble(0.0),
    thrOverEEB1 = cms.vdouble(0.0),
    thrOverEEB2 = cms.vdouble(0.0),
    thrOverEEE1 = cms.vdouble(0.0),
    thrOverEEE2 = cms.vdouble(0.0),
    thrRegularEB1 = cms.vdouble(0.02),
    thrRegularEB2 = cms.vdouble(0.09),
    thrRegularEE1 = cms.vdouble(0.04),
    thrRegularEE2 = cms.vdouble(0.04),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaGsfTrackVarsL1Seeded","Dphi")
)

hltEle30WPTightBestGsfNLayerITL1SeededFilter = _hltEle32WPTightBestGsfNLayerITL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag("hltEle30WPTightGsfDphiL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(False),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(0),
    thrOverE2EE = cms.vdouble(0),
    thrOverEEB = cms.vdouble(0),
    thrOverEEE = cms.vdouble(0),
    thrRegularEB = cms.vdouble(3),
    thrRegularEE = cms.vdouble(3),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaBestGsfTrackVarsL1Seeded","NLayerIT")
)

hltEle30WPTightBestGsfChi2L1SeededFilter = _hltEle32WPTightBestGsfChi2L1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    candTag = cms.InputTag("hltEle30WPTightBestGsfNLayerITL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag(""),
    saveTags = cms.bool(True),
    thrOverE2EB = cms.vdouble(0),
    thrOverE2EE = cms.vdouble(0),
    thrOverEEB = cms.vdouble(0),
    thrOverEEE = cms.vdouble(0),
    thrRegularEB = cms.vdouble(50.0),
    thrRegularEE = cms.vdouble(50.0),
    useEt = cms.bool(False),
    varTag = cms.InputTag("hltEgammaBestGsfTrackVarsL1Seeded","Chi2")
)

hltEle30WPTightGsfTrackIsoFromL1TracksL1SeededFilter = _hltEle32WPTightGsfTrackIsoFromL1TracksL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 0.8, 1.479, 2.0),
    candTag = cms.InputTag("hltEle30WPTightBestGsfChi2L1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.0, 0.0, 0.0, 0.0),
    energyLowEdges = cms.vdouble(0.0),
    etaBoundaryEB12 = cms.double(0.8),
    etaBoundaryEE12 = cms.double(2.0),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag("hltFixedGridRhoFastjetAllCaloForEGamma"),
    saveTags = cms.bool(True),
    thrOverE2EB1 = cms.vdouble(0.0),
    thrOverE2EB2 = cms.vdouble(0.0),
    thrOverE2EE1 = cms.vdouble(0.0),
    thrOverE2EE2 = cms.vdouble(0.0),
    thrOverEEB1 = cms.vdouble(0.0),
    thrOverEEB2 = cms.vdouble(0.0),
    thrOverEEE1 = cms.vdouble(0.0),
    thrOverEEE2 = cms.vdouble(0.0),
    thrRegularEB1 = cms.vdouble(5.5),
    thrRegularEB2 = cms.vdouble(8.0),
    thrRegularEE1 = cms.vdouble(5.5),
    thrRegularEE2 = cms.vdouble(5.5),
    useEt = cms.bool(True),
    varTag = cms.InputTag("hltEgammaEleL1TrkIsoL1Seeded")
)

hltEle30WPTightGsfTrackIsoL1SeededFilter = _hltEle32WPTightGsfTrackIsoL1SeededFilter.clone(
    absEtaLowEdges = cms.vdouble(0.0, 1.0, 1.479, 2.1),
    candTag = cms.InputTag("hltEle30WPTightGsfTrackIsoFromL1TracksL1SeededFilter"),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.029, 0.111, 0.114, 0.032),
    energyLowEdges = cms.vdouble(0.0),
    etaBoundaryEB12 = cms.double(1.0),
    etaBoundaryEE12 = cms.double(2.1),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    lessThan = cms.bool(True),
    ncandcut = cms.int32(1),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    rhoTag = cms.InputTag("hltFixedGridRhoFastjetAllCaloForEGamma"),
    saveTags = cms.bool(True),
    thrOverE2EB1 = cms.vdouble(0.0),
    thrOverE2EB2 = cms.vdouble(0.0),
    thrOverE2EE1 = cms.vdouble(0.0),
    thrOverE2EE2 = cms.vdouble(0.0),
    thrOverEEB1 = cms.vdouble(0.0),
    thrOverEEB2 = cms.vdouble(0.0),
    thrOverEEE1 = cms.vdouble(0.0),
    thrOverEEE2 = cms.vdouble(0.0),
    thrRegularEB1 = cms.vdouble(2.5),
    thrRegularEB2 = cms.vdouble(2.5),
    thrRegularEE1 = cms.vdouble(2.2),
    thrRegularEE2 = cms.vdouble(2.2),
    useEt = cms.bool(True),
    varTag = cms.InputTag("hltEgammaEleGsfTrackIsoV6L1Seeded")
)


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
