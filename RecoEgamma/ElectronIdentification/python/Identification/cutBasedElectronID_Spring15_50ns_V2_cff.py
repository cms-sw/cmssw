from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_tools import *

#
# This is the first round of Spring15 50ns cuts, optimized on  Spring15 50ns samples. 
#
# The ID cuts below are optimized IDs for Spring15 Scenario with 50ns bunch spacing
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
#        https://indico.cern.ch/event/369245/contribution/2/attachments/1153005/1655984/Rami_eleCB_ID_50ns_V2.pdf
# NOTE: this V2 version is different from the V1 version for these conditions only
# by the cuts of the WP Tight for the barrel. All other WP and endcap are the same as V1.
# The changes was needed to make the WP Tight EB tighter than the HLT. For reference, the full V1 talk is here:
#        https://indico.cern.ch/event/369239/contribution/6/attachments/1134836/1623383/Rami_eleCB_ID_50ns.pdf
#
# First, define cut values
#

# Veto working point Barrel and Endcap
idName = "cutBasedElectronID-Spring15-50ns-V2-standalone-veto"
WP_Veto_EB = EleWorkingPoint_V2(
    idName   , # idName
    0.0126   , # dEtaInCut
    0.107    , # dPhiInCut
    0.012    , # full5x5_sigmaIEtaIEtaCut
    0.186    , # hOverECut
    0.0621   , # dxyCut
    0.613    , # dzCut
    0.239    , # absEInverseMinusPInverseCut
    0.161    , # relCombIsolationWithEALowPtCut
    0.161 , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    2          # missingHitsCut
    )

WP_Veto_EE = EleWorkingPoint_V2(
    idName   , # idName
    0.0109   , # dEtaInCut
    0.219    , # dPhiInCut
    0.0339   , # full5x5_sigmaIEtaIEtaCut
    0.0962   , # hOverECut
    0.279    , # dxyCut
    0.947    , # dzCut
    0.141    , # absEInverseMinusPInverseCut
    0.193    , # relCombIsolationWithEALowPtCut
    0.193    , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    3          # missingHitsCut
    )

# Loose working point Barrel and Endcap
idName = "cutBasedElectronID-Spring15-50ns-V2-standalone-loose"
WP_Loose_EB = EleWorkingPoint_V2(
    idName   , # idName
    0.0098   , # dEtaInCut
    0.0929   , # dPhiInCut
    0.0105   , # full5x5_sigmaIEtaIEtaCut
    0.0765   , # hOverECut
    0.0227   , # dxyCut
    0.379    , # dzCut
    0.184    , # absEInverseMinusPInverseCut
    0.118    , # relCombIsolationWithEALowPtCut
    0.118    , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    2          # missingHitsCut
    )

WP_Loose_EE = EleWorkingPoint_V2(
    idName   , # idName
    0.00950  , # dEtaInCut
    0.181    , # dPhiInCut
    0.0318   , # full5x5_sigmaIEtaIEtaCut
    0.0824   , # hOverECut
    0.242    , # dxyCut
    0.921    , # dzCut
    0.125    , # absEInverseMinusPInverseCut
    0.118    , # relCombIsolationWithEALowPtCut
    0.118    , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

# Medium working point Barrel and Endcap
idName = "cutBasedElectronID-Spring15-50ns-V2-standalone-medium"
WP_Medium_EB = EleWorkingPoint_V2(
    idName   , # idName
    0.00945  , # dEtaInCut
    0.0296   , # dPhiInCut
    0.0101   , # full5x5_sigmaIEtaIEtaCut
    0.0372   , # hOverECut
    0.0151   , # dxyCut
    0.238    , # dzCut
    0.118    , # absEInverseMinusPInverseCut
    0.0987   , # relCombIsolationWithEALowPtCut
    0.0987   , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    2          # missingHitsCut
    )

WP_Medium_EE = EleWorkingPoint_V2(
    idName   , # idName
    0.00773  , # dEtaInCut
    0.148    , # dPhiInCut
    0.0287   , # full5x5_sigmaIEtaIEtaCut
    0.0546   , # hOverECut
    0.0535   , # dxyCut
    0.572    , # dzCut
    0.104    , # absEInverseMinusPInverseCut
    0.0902   , # relCombIsolationWithEALowPtCut
    0.0902   , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

# Tight working point Barrel and Endcap
idName = "cutBasedElectronID-Spring15-50ns-V2-standalone-tight"
WP_Tight_EB = EleWorkingPoint_V2(
    idName   , # idName
    0.00864  , # dEtaInCut
    0.0286   , # dPhiInCut
    0.0101   , # full5x5_sigmaIEtaIEtaCut
    0.0342   , # hOverECut
    0.0103   , # dxyCut
    0.170    , # dzCut
    0.0116   , # absEInverseMinusPInverseCut
    0.0591   , # relCombIsolationWithEALowPtCut
    0.0591   , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    2          # missingHitsCut
    )

WP_Tight_EE = EleWorkingPoint_V2(
    idName   , # idName
    0.00762  , # dEtaInCut
    0.0439   , # dPhiInCut
    0.0287   , # full5x5_sigmaIEtaIEtaCut
    0.0544   , # hOverECut
    0.0377   , # dxyCut
    0.571    , # dzCut
    0.0100   , # absEInverseMinusPInverseCut
    0.0759   , # relCombIsolationWithEALowPtCut
    0.0759   , # relCombIsolationWithEAHighPtCut
    # conversion veto cut needs no parameters, so not mentioned
    1          # missingHitsCut
    )

# Second, define what effective areas to use for pile-up correction
isoEffAreas = "RecoEgamma/ElectronIdentification/data/Spring15/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_50ns.txt"


#
# Set up VID configuration for all cuts and working points
#

cutBasedElectronID_Spring15_50ns_V2_standalone_veto = configureVIDCutBasedEleID_V2(WP_Veto_EB, WP_Veto_EE, isoEffAreas)
cutBasedElectronID_Spring15_50ns_V2_standalone_loose = configureVIDCutBasedEleID_V2(WP_Loose_EB, WP_Loose_EE, isoEffAreas)
cutBasedElectronID_Spring15_50ns_V2_standalone_medium = configureVIDCutBasedEleID_V2(WP_Medium_EB, WP_Medium_EE, isoEffAreas)
cutBasedElectronID_Spring15_50ns_V2_standalone_tight = configureVIDCutBasedEleID_V2(WP_Tight_EB, WP_Tight_EE, isoEffAreas)

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedElectronID_Spring15_50ns_V2_standalone_veto.idName,
                             '3edb92d2aee3dcf1e366a927a5660155')
central_id_registry.register(cutBasedElectronID_Spring15_50ns_V2_standalone_loose.idName,
                             '527c1b1ddcf9061dc4093dc95590e2bb')
central_id_registry.register(cutBasedElectronID_Spring15_50ns_V2_standalone_medium.idName,
                             '6837f1ac82974f19d2a15041a2e52ebb')
central_id_registry.register(cutBasedElectronID_Spring15_50ns_V2_standalone_tight.idName,
                             'ab050ff2bd30d832881bfac03c9d3a8a')


# for now until we have a database...
cutBasedElectronID_Spring15_50ns_V2_standalone_veto.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_Spring15_50ns_V2_standalone_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_Spring15_50ns_V2_standalone_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_Spring15_50ns_V2_standalone_tight.isPOGApproved = cms.untracked.bool(True)
