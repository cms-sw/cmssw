from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_tools import *

#
# This is the first second of PHYS14 cuts, optimized on  PHYS14 samples. 
#
# The ID cuts below are optimized IDs for PHYS14 Scenario PU 20, bx 25ns
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
#        https://indico.cern.ch/event/370494/contribution/2/material/slides/0.pdf
#
# First, define cut values
#

# Veto working point Barrel and Endcap
idName = "cutBasedElectronID-PHYS14-PU20bx25-V2-standalone-veto"
WP_Veto_EB = EleWorkingPoint_V2(
    idName   , # idName
    0.013625 , # dEtaInCut
    0.230374 , # dPhiInCut
    0.011586 , # full5x5_sigmaIEtaIEtaCut
    0.181130 , # hOverECut
    0.094095 , # dxyCut
    0.713070 , # dzCut
    0.295751 , # absEInverseMinusPInverseCut
    0.158721 , # relCombIsolationWithEALowPtCut
    0.158721 , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    2          # missingHitsCut
    )

WP_Veto_EE = EleWorkingPoint_V2(
    idName   , # idName
    0.011932 , # dEtaInCut
    0.255450 , # dPhiInCut
    0.031849 , # full5x5_sigmaIEtaIEtaCut
    0.223870 , # hOverECut
    0.342293 , # dxyCut
    0.953461 , # dzCut
    0.155501 , # absEInverseMinusPInverseCut
    0.177032 , # relCombIsolationWithEALowPtCut
    0.177032 , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    3          # missingHitsCut
    )

# Loose working point Barrel and Endcap
idName = "cutBasedElectronID-PHYS14-PU20bx25-V2-standalone-loose"
WP_Loose_EB = EleWorkingPoint_V2(
    idName   , # idName
    0.009277 , # dEtaInCut
    0.094739 , # dPhiInCut
    0.010331 , # full5x5_sigmaIEtaIEtaCut
    0.093068 , # hOverECut
    0.035904 , # dxyCut
    0.075496 , # dzCut
    0.189968 , # absEInverseMinusPInverseCut
    0.130136 , # relCombIsolationWithEALowPtCut
    0.130136 , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

WP_Loose_EE = EleWorkingPoint_V2(
    idName   , # idName
    0.009833 , # dEtaInCut
    0.149934 , # dPhiInCut
    0.031838 , # full5x5_sigmaIEtaIEtaCut
    0.115754 , # hOverECut
    0.099266 , # dxyCut
    0.197897 , # dzCut
    0.140662 , # absEInverseMinusPInverseCut
    0.163368 , # relCombIsolationWithEALowPtCut
    0.163368 , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

# Medium working point Barrel and Endcap
idName = "cutBasedElectronID-PHYS14-PU20bx25-V2-standalone-medium"
WP_Medium_EB = EleWorkingPoint_V2(
    idName   , # idName
    0.008925 , # dEtaInCut
    0.035973 , # dPhiInCut
    0.009996 , # full5x5_sigmaIEtaIEtaCut
    0.050537 , # hOverECut
    0.012235 , # dxyCut
    0.042020 , # dzCut
    0.091942 , # absEInverseMinusPInverseCut
    0.107587 , # relCombIsolationWithEALowPtCut
    0.107587 , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

WP_Medium_EE = EleWorkingPoint_V2(
    idName   , # idName
    0.007429 , # dEtaInCut
    0.067879 , # dPhiInCut
    0.030135 , # full5x5_sigmaIEtaIEtaCut
    0.086782 , # hOverECut
    0.036719 , # dxyCut
    0.138142 , # dzCut
    0.100683 , # absEInverseMinusPInverseCut
    0.113254 , # relCombIsolationWithEALowPtCut
    0.113254 , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

# Tight working point Barrel and Endcap
idName = "cutBasedElectronID-PHYS14-PU20bx25-V2-standalone-tight"
WP_Tight_EB = EleWorkingPoint_V2(
    idName   , # idName
    0.006046 , # dEtaInCut
    0.028092 , # dPhiInCut
    0.009947 , # full5x5_sigmaIEtaIEtaCut
    0.045772 , # hOverECut
    0.008790 , # dxyCut
    0.021226 , # dzCut
    0.020118 , # absEInverseMinusPInverseCut
    0.069537 , # relCombIsolationWithEALowPtCut
    0.069537 , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

WP_Tight_EE = EleWorkingPoint_V2(
    idName   , # idName
    0.007057 , # dEtaInCut
    0.030159 , # dPhiInCut
    0.028237 , # full5x5_sigmaIEtaIEtaCut
    0.067778 , # hOverECut
    0.027984 , # dxyCut
    0.133431 , # dzCut
    0.098919 , # absEInverseMinusPInverseCut
    0.078265 , # relCombIsolationWithEALowPtCut
    0.078265 , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

# Second, define what effective areas to use for pile-up correction
isoInputs = IsolationCutInputs_V2(
    # phoIsolationEffAreas
    "RecoEgamma/ElectronIdentification/data/PHYS14/effAreaElectrons_cone03_pfNeuHadronsAndPhotons.txt"
)


#
# Set up VID configuration for all cuts and working points
#

cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_veto = configureVIDCutBasedEleID_V2(WP_Veto_EB, WP_Veto_EE, isoInputs)
cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_loose = configureVIDCutBasedEleID_V2(WP_Loose_EB, WP_Loose_EE, isoInputs)
cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_medium = configureVIDCutBasedEleID_V2(WP_Medium_EB, WP_Medium_EE, isoInputs)
cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_tight = configureVIDCutBasedEleID_V2(WP_Tight_EB, WP_Tight_EE, isoInputs)


central_id_registry.register(cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_veto.idName,
                             '845f6b06a607cf0ab02a136909de6fdc')
central_id_registry.register(cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_loose.idName,
                             '0ccc07e6287048b3fccdc4d37d2ceb95')
central_id_registry.register(cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_medium.idName,
                             '7c5b51d5072d760cf19ca4f3ac5c65f3')
central_id_registry.register(cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_tight.idName,
                             '367f3260b6ff2a60ef76b2dc6f1ebd16')


# for now until we have a database...
cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_veto.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_PHYS14_PU20bx25_V2_standalone_tight.isPOGApproved = cms.untracked.bool(True)
