from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_tools \
    import ( EleWorkingPoint_V5,
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
#122X V1 IDs for Run3(first set of IDs for Run3) 
idName = "cutBasedElectronID-Fall22-122X-V1-veto"
WP_Veto_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0117  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.0071  , # dEtaInSeedCut
    dPhiInCut                      = 0.208   , # dPhiInCut
    hOverECut_C0                   = 0.05    , # hOverECut
    hOverECut_CE                   = 1.28    ,
    hOverECut_Cr                   = 0.0422  ,
    relCombIsolationWithEACut_C0   = 0.406   , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.535   ,
    absEInverseMinusPInverseCut    = 0.178   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 2          # missingHitsCut
    )

WP_Veto_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0298  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.0173  , # dEtaInSeedCut
    dPhiInCut                      = 0.234   , # dPhiInCut
    hOverECut_C0                   = 0.05    , # hOverECut
    hOverECut_CE                   = 2.3     ,
    hOverECut_Cr                   = 0.262   ,
    relCombIsolationWithEACut_C0   = 0.342   , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.519   ,
    absEInverseMinusPInverseCut    = 0.137   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 3          # missingHitsCut
    )

# Loose working point Barrel and Endcap
idName = "cutBasedElectronID-Fall22-122X-V1-loose"
WP_Loose_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0107  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00691 , # dEtaInSeedCut
    dPhiInCut                      = 0.175   , # dPhiInCut
    hOverECut_C0                   = 0.05    , # hOverECut
    hOverECut_CE                   = 1.28    ,
    hOverECut_Cr                   = 0.0422  ,
    relCombIsolationWithEACut_C0   = 0.194   , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.535   ,
    absEInverseMinusPInverseCut    = 0.138   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Loose_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0275  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.0121  , # dEtaInSeedCut
    dPhiInCut                      = 0.228   , # dPhiInCut
    hOverECut_C0                   = 0.05    , # hOverECut
    hOverECut_CE                   = 2.3     ,
    hOverECut_Cr                   = 0.262   ,
    relCombIsolationWithEACut_C0   = 0.184   , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.519   ,
    absEInverseMinusPInverseCut    = 0.127   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1         # missingHitsCut
    )

# Medium working point Barrel and Endcap
idName = "cutBasedElectronID-Fall22-122X-V1-medium"
WP_Medium_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0103  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00481 , # dEtaInSeedCut
    dPhiInCut                      = 0.127   , # dPhiInCut
    hOverECut_C0                   = 0.0241  , # hOverECut
    hOverECut_CE                   = 1.28    ,
    hOverECut_Cr                   = 0.0422  ,
    relCombIsolationWithEACut_C0   = 0.0837  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.535   ,
    absEInverseMinusPInverseCut    = 0.0966  , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Medium_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0272  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00951 , # dEtaInSeedCut
    dPhiInCut                      = 0.221   , # dPhiInCut
    hOverECut_C0                   = 0.05    , # hOverECut
    hOverECut_CE                   = 2.3     ,
    hOverECut_Cr                   = 0.262   ,
    relCombIsolationWithEACut_C0   = 0.0741  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.519   ,
    absEInverseMinusPInverseCut    = 0.111  , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

# Tight working point Barrel and Endcap
idName = "cutBasedElectronID-Fall22-122X-V1-tight"
WP_Tight_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0101  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00411 , # dEtaInSeedCut
    dPhiInCut                      = 0.116   , # dPhiInCut
    hOverECut_C0                   = 0.02    , # hOverECut
    hOverECut_CE                   = 1.16    ,
    hOverECut_Cr                   = 0.0422  ,
    relCombIsolationWithEACut_C0   = 0.0388  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.535   ,
    absEInverseMinusPInverseCut    = 0.023   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Tight_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.027   , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00938 , # dEtaInSeedCut
    dPhiInCut                      = 0.164   , # dPhiInCut
    hOverECut_C0                   = 0.02    , # hOverECut
    hOverECut_CE                   = 0.5     ,
    hOverECut_Cr                   = 0.262   ,
    relCombIsolationWithEACut_C0   = 0.0544  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.519   ,
    absEInverseMinusPInverseCut    = 0.018   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

# Second, define what effective areas to use for pile-up correction
#isoEffAreas = "RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"
isoEffAreas = "RecoEgamma/ElectronIdentification/data/Run3_Fall22/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_122X.txt"
#
# Set up VID configuration for all cuts and working points
#
cutBasedElectronID_Fall22_122X_V1_veto   = configureVIDCutBasedEleID_V5(WP_Veto_EB,   WP_Veto_EE, isoEffAreas)
cutBasedElectronID_Fall22_122X_V1_loose  = configureVIDCutBasedEleID_V5(WP_Loose_EB,  WP_Loose_EE, isoEffAreas)
cutBasedElectronID_Fall22_122X_V1_medium = configureVIDCutBasedEleID_V5(WP_Medium_EB, WP_Medium_EE, isoEffAreas)
cutBasedElectronID_Fall22_122X_V1_tight  = configureVIDCutBasedEleID_V5(WP_Tight_EB,  WP_Tight_EE, isoEffAreas)

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry and the isPOGApproved lines,
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#
central_id_registry.register(cutBasedElectronID_Fall22_122X_V1_veto.idName,   'f2bdd2bd67779f62ee94df103c8632efa0e4e9e3')
central_id_registry.register(cutBasedElectronID_Fall22_122X_V1_loose.idName,  '002bb55c0a6176fa07ffce0672a5e82843e83738')
central_id_registry.register(cutBasedElectronID_Fall22_122X_V1_medium.idName, '6eeb1197bf3a564d089c6a8213c895292f97de02')
central_id_registry.register(cutBasedElectronID_Fall22_122X_V1_tight.idName,  '715df1da203dff03f39a347f658b2471472120d9')

### for now until we have a database...
cutBasedElectronID_Fall22_122X_V1_veto.isPOGApproved   = cms.untracked.bool(True)
cutBasedElectronID_Fall22_122X_V1_loose.isPOGApproved  = cms.untracked.bool(True)
cutBasedElectronID_Fall22_122X_V1_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_Fall22_122X_V1_tight.isPOGApproved  = cms.untracked.bool(True)

### 94X_v2 registry values
#central_id_registry.register(cutBasedElectronID_Fall17_94X_V2_veto.idName,   '74e217e3ece16b49bd337026a29fc3e9')
#central_id_registry.register(cutBasedElectronID_Fall17_94X_V2_loose.idName,  '5547e2c8b5c222192519c41bff05bc2e')
#central_id_registry.register(cutBasedElectronID_Fall17_94X_V2_medium.idName, '48702f025a8df2c527f53927af8b66d0')
#central_id_registry.register(cutBasedElectronID_Fall17_94X_V2_tight.idName,  'c06761e199f084f5b0f7868ac48a3e19')
