from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_tools import *

#
# This is the first version of PHYS14 cuts, optimized on  PHYS14 samples. 
#
# The ID cuts below are optimized IDs for PHYS14 Scenario PU 20, bx 25ns
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
#        https://indico.cern.ch/event/292938/contribution/0/material/slides/1.pdf

#
# First, define cut values
#

# Veto working point Barrel and Endcap
idName = "cutBasedElectronID-PHYS14-PU20bx25-V1-standalone-veto"
WP_Veto_EB = EleWorkingPoint_V1(
    idName   , # idName
    0.016315 , # dEtaInCut
    0.252044 , # dPhiInCut
    0.011100 , # full5x5_sigmaIEtaIEtaCut
    0.345843 , # hOverECut
    0.060279 , # dxyCut
    0.800538 , # dzCut
    0.248070 , # absEInverseMinusPInverseCut
    0.164369 , # relCombIsolationWithDBetaLowPtCut
    0.164369 , # relCombIsolationWithDBetaHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    2          # missingHitsCut
    )

WP_Veto_EE = EleWorkingPoint_V1(
    idName   , # idName
    0.010671 , # dEtaInCut
    0.245263 , # dPhiInCut
    0.033987 , # full5x5_sigmaIEtaIEtaCut
    0.134691 , # hOverECut
    0.273097 , # dxyCut
    0.885860 , # dzCut
    0.157160 , # absEInverseMinusPInverseCut
    0.212604 , # relCombIsolationWithDBetaLowPtCut
    0.212604 , # relCombIsolationWithDBetaHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    3          # missingHitsCut
    )

# Loose working point Barrel and Endcap
idName = "cutBasedElectronID-PHYS14-PU20bx25-V1-standalone-loose"
WP_Loose_EB = EleWorkingPoint_V1(
    idName   , # idName
    0.012442 , # dEtaInCut
    0.072624 , # dPhiInCut
    0.010557 , # full5x5_sigmaIEtaIEtaCut
    0.121476 , # hOverECut
    0.022664 , # dxyCut
    0.173670 , # dzCut
    0.221803 , # absEInverseMinusPInverseCut
    0.120026 , # relCombIsolationWithDBetaLowPtCut
    0.120026 , # relCombIsolationWithDBetaHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

WP_Loose_EE = EleWorkingPoint_V1(
    idName   , # idName
    0.010654 , # dEtaInCut
    0.145129 , # dPhiInCut
    0.032602 , # full5x5_sigmaIEtaIEtaCut
    0.131862 , # hOverECut
    0.097358 , # dxyCut
    0.198444 , # dzCut
    0.142283 , # absEInverseMinusPInverseCut
    0.162914 , # relCombIsolationWithDBetaLowPtCut
    0.162914 , # relCombIsolationWithDBetaHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

# Medium working point Barrel and Endcap
idName = "cutBasedElectronID-PHYS14-PU20bx25-V1-standalone-medium"
WP_Medium_EB = EleWorkingPoint_V1(
    idName   , # idName
    0.007641 , # dEtaInCut
    0.032643 , # dPhiInCut
    0.010399 , # full5x5_sigmaIEtaIEtaCut
    0.060662 , # hOverECut
    0.011811 , # dxyCut
    0.070775 , # dzCut
    0.153897 , # absEInverseMinusPInverseCut
    0.097213 , # relCombIsolationWithDBetaLowPtCut
    0.097213 , # relCombIsolationWithDBetaHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

WP_Medium_EE = EleWorkingPoint_V1(
    idName   , # idName
    0.009285 , # dEtaInCut
    0.042447 , # dPhiInCut
    0.029524 , # full5x5_sigmaIEtaIEtaCut
    0.104263 , # hOverECut
    0.051682 , # dxyCut
    0.180720 , # dzCut
    0.137468 , # absEInverseMinusPInverseCut
    0.116708 , # relCombIsolationWithDBetaLowPtCut
    0.116708 , # relCombIsolationWithDBetaHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

# Tight working point Barrel and Endcap
idName = "cutBasedElectronID-PHYS14-PU20bx25-V1-standalone-tight"
WP_Tight_EB = EleWorkingPoint_V1(
    idName   , # idName
    0.006574 , # dEtaInCut
    0.022868 , # dPhiInCut
    0.010181 , # full5x5_sigmaIEtaIEtaCut
    0.037553 , # hOverECut
    0.009924 , # dxyCut
    0.015310 , # dzCut
    0.131191 , # absEInverseMinusPInverseCut
    0.074355 , # relCombIsolationWithDBetaLowPtCut
    0.074355 , # relCombIsolationWithDBetaHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

WP_Tight_EE = EleWorkingPoint_V1(
    idName   , # idName
    0.005681 , # dEtaInCut
    0.032046 , # dPhiInCut
    0.028766 , # full5x5_sigmaIEtaIEtaCut
    0.081902 , # hOverECut
    0.027261 , # dxyCut
    0.147154 , # dzCut
    0.106055 , # absEInverseMinusPInverseCut
    0.090185 , # relCombIsolationWithDBetaLowPtCut
    0.090185 , # relCombIsolationWithDBetaHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

#
# Set up VID configuration for all cuts and working points
#

cutBasedElectronID_PHYS14_PU20bx25_V1_standalone_veto = configureVIDCutBasedEleID_V1(WP_Veto_EB, WP_Veto_EE)
cutBasedElectronID_PHYS14_PU20bx25_V1_standalone_loose = configureVIDCutBasedEleID_V1(WP_Loose_EB, WP_Loose_EE)
cutBasedElectronID_PHYS14_PU20bx25_V1_standalone_medium = configureVIDCutBasedEleID_V1(WP_Medium_EB, WP_Medium_EE)
cutBasedElectronID_PHYS14_PU20bx25_V1_standalone_tight = configureVIDCutBasedEleID_V1(WP_Tight_EB, WP_Tight_EE)


central_id_registry.register(cutBasedElectronID_PHYS14_PU20bx25_V1_standalone_veto.idName,
                             '23182e502012ea0c88986a0a6dae2ade')
central_id_registry.register(cutBasedElectronID_PHYS14_PU20bx25_V1_standalone_loose.idName,
                             '1ac79d48189ade31d89a2acc91f72cbf')
central_id_registry.register(cutBasedElectronID_PHYS14_PU20bx25_V1_standalone_medium.idName,
                             '08f53e341990e959ecf8e24cb0214f55')
central_id_registry.register(cutBasedElectronID_PHYS14_PU20bx25_V1_standalone_tight.idName,
                             '93e5e30a36f1fbdce074c110463481ab')
