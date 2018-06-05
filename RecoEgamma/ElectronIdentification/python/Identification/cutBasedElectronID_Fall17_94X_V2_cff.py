from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_tools \
    import ( EleWorkingPoint_V5,
             IsolationCutInputs_V2,
             configureVIDCutBasedEleID_V5 )

#
# The ID cuts below are optimized IDs on Fall17 simulation with 94X-based production
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
#  https://indico.cern.ch/event/697079/ 
#
#

# Veto working point Barrel and Endcap
#V2 of IDs good for Moriond 18
idName = "cutBasedElectronID-Fall17-94X-V2-veto"
WP_Veto_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0126  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00463 , # dEtaInSeedCut
    dPhiInCut                      = 0.148   , # dPhiInCut
    hOverECut_C0                   = 0.05    , # hOverECut
    hOverECut_CE                   = 1.16    ,
    hOverECut_Cr                   = 0.0324  ,
    relCombIsolationWithEACut_C0   = 0.198   , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.506   ,
    absEInverseMinusPInverseCut    = 0.209   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 2          # missingHitsCut
    )

WP_Veto_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0457  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00814 , # dEtaInSeedCut
    dPhiInCut                      = 0.19    , # dPhiInCut
    hOverECut_C0                   = 0.05    , # hOverECut
    hOverECut_CE                   = 2.54    ,
    hOverECut_Cr                   = 0.183   ,
    relCombIsolationWithEACut_C0   = 0.203   , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.963   ,
    absEInverseMinusPInverseCut    = 0.132   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 3          # missingHitsCut
    )

# Loose working point Barrel and Endcap
idName = "cutBasedElectronID-Fall17-94X-V2-loose"
WP_Loose_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0112  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00377 , # dEtaInSeedCut
    dPhiInCut                      = 0.0884  , # dPhiInCut
    hOverECut_C0                   = 0.05    , # hOverECut
    hOverECut_CE                   = 1.16    ,
    hOverECut_Cr                   = 0.0324  ,
    relCombIsolationWithEACut_C0   = 0.112   , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.506   ,
    absEInverseMinusPInverseCut    = 0.193   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Loose_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0425  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00674 , # dEtaInSeedCut
    dPhiInCut                      = 0.169   , # dPhiInCut
    hOverECut_C0                   = 0.0441  , # hOverECut
    hOverECut_CE                   = 2.54    ,
    hOverECut_Cr                   = 0.183   ,
    relCombIsolationWithEACut_C0   = 0.108   , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.963   ,
    absEInverseMinusPInverseCut    = 0.111   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1         # missingHitsCut
    )

# Medium working point Barrel and Endcap
idName = "cutBasedElectronID-Fall17-94X-V2-medium"
WP_Medium_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0106  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.0032  , # dEtaInSeedCut
    dPhiInCut                      = 0.0547  , # dPhiInCut
    hOverECut_C0                   = 0.046   , # hOverECut
    hOverECut_CE                   = 1.16    ,
    hOverECut_Cr                   = 0.0324  ,
    relCombIsolationWithEACut_C0   = 0.0478  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.506   ,
    absEInverseMinusPInverseCut    = 0.184   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Medium_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0387  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00632 , # dEtaInSeedCut
    dPhiInCut                      = 0.0394  , # dPhiInCut
    hOverECut_C0                   = 0.0275  , # hOverECut
    hOverECut_CE                   = 2.52    ,
    hOverECut_Cr                   = 0.183   ,
    relCombIsolationWithEACut_C0   = 0.0658  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.963   ,
    absEInverseMinusPInverseCut    = 0.0721  , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

# Tight working point Barrel and Endcap
idName = "cutBasedElectronID-Fall17-94X-V2-tight"
WP_Tight_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0104  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00255 , # dEtaInSeedCut
    dPhiInCut                      = 0.022   , # dPhiInCut
    hOverECut_C0                   = 0.026   , # hOverECut
    hOverECut_CE                   = 1.15    ,
    hOverECut_Cr                   = 0.0324  ,
    relCombIsolationWithEACut_C0   = 0.0287  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.506   ,
    absEInverseMinusPInverseCut    = 0.159   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Tight_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0353  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00501 , # dEtaInSeedCut
    dPhiInCut                      = 0.0236  , # dPhiInCut
    hOverECut_C0                   = 0.0188  , # hOverECut
    hOverECut_CE                   = 2.06    ,
    hOverECut_Cr                   = 0.183   ,
    relCombIsolationWithEACut_C0   = 0.0445  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.963   ,
    absEInverseMinusPInverseCut    = 0.0197  , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

# Second, define what effective areas to use for pile-up correction
isoInputs = IsolationCutInputs_V2(
    # phoIsolationEffAreas
    "RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"
)


#
# Set up VID configuration for all cuts and working points
#

cutBasedElectronID_Fall17_94X_V2_veto   = configureVIDCutBasedEleID_V5(WP_Veto_EB,   WP_Veto_EE, isoInputs)
cutBasedElectronID_Fall17_94X_V2_loose  = configureVIDCutBasedEleID_V5(WP_Loose_EB,  WP_Loose_EE, isoInputs)
cutBasedElectronID_Fall17_94X_V2_medium = configureVIDCutBasedEleID_V5(WP_Medium_EB, WP_Medium_EE, isoInputs)
cutBasedElectronID_Fall17_94X_V2_tight  = configureVIDCutBasedEleID_V5(WP_Tight_EB,  WP_Tight_EE, isoInputs)


# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

#central_id_registry.register(cutBasedElectronID_Fall17_94X_V1_veto.idName,   '43be9b381a8d9b0910b7f81a5ad8ff3a')
#central_id_registry.register(cutBasedElectronID_Fall17_94X_V1_loose.idName,  '0b8456d622494441fe713a6858e0f7c1')
#central_id_registry.register(cutBasedElectronID_Fall17_94X_V1_medium.idName, 'a238ee70910de53d36866e89768500e9')
#central_id_registry.register(cutBasedElectronID_Fall17_94X_V1_tight.idName,  '4acb2d2796efde7fba75380ce8823fc2')

### for now until we have a database...
cutBasedElectronID_Fall17_94X_V2_veto.isPOGApproved   = cms.untracked.bool(True)
cutBasedElectronID_Fall17_94X_V2_loose.isPOGApproved  = cms.untracked.bool(True)
cutBasedElectronID_Fall17_94X_V2_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_Fall17_94X_V2_tight.isPOGApproved  = cms.untracked.bool(True)
