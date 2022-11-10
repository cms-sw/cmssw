from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_tools \
    import ( EleWorkingPoint_V5,
             configureVIDCutBasedEleID_V5 )

#
# The ID cuts below are optimized IDs on Winter22 simulation with 122X-based production
# The cut values and the ID optimization discussions can be found at:
# https://indico.cern.ch/event/1204275/contributions/5064343/attachments/2529616/4353987/Electron_cutbasedID_preliminaryID.pdf
#
#

# Veto working point Barrel and Endcap
#Winter22_122X V1 IDs for Run3(first set of IDs for Run3) 
idName = "cutBasedElectronID-RunIIIWinter22-V1-veto"
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
idName = "cutBasedElectronID-RunIIIWinter22-V1-loose"
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
idName = "cutBasedElectronID-RunIIIWinter22-V1-medium"
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
idName = "cutBasedElectronID-RunIIIWinter22-V1-tight"
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
isoEffAreas = "RecoEgamma/ElectronIdentification/data/Run3_Winter22/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_122X.txt"
#
# Set up VID configuration for all cuts and working points
#
cutBasedElectronID_RunIIIWinter22_V1_veto   = configureVIDCutBasedEleID_V5(WP_Veto_EB,   WP_Veto_EE, isoEffAreas)
cutBasedElectronID_RunIIIWinter22_V1_loose  = configureVIDCutBasedEleID_V5(WP_Loose_EB,  WP_Loose_EE, isoEffAreas)
cutBasedElectronID_RunIIIWinter22_V1_medium = configureVIDCutBasedEleID_V5(WP_Medium_EB, WP_Medium_EE, isoEffAreas)
cutBasedElectronID_RunIIIWinter22_V1_tight  = configureVIDCutBasedEleID_V5(WP_Tight_EB,  WP_Tight_EE, isoEffAreas)

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry and the isPOGApproved lines,
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#
central_id_registry.register(cutBasedElectronID_RunIIIWinter22_V1_veto.idName,   '04d495d199252c2017d5019ae8b478a7d8aebc79')
central_id_registry.register(cutBasedElectronID_RunIIIWinter22_V1_loose.idName,  '648b0cc1957047ffe3f027111389dcf5aa941edc')
central_id_registry.register(cutBasedElectronID_RunIIIWinter22_V1_medium.idName, '2626edc1ad1dc1673c0713c557df78f3e90a66f5')
central_id_registry.register(cutBasedElectronID_RunIIIWinter22_V1_tight.idName,  '2331bfa0b099f80090aa1d48df03b7a134cf788e')

### for now until we have a database...
cutBasedElectronID_RunIIIWinter22_V1_veto.isPOGApproved   = cms.untracked.bool(True)
cutBasedElectronID_RunIIIWinter22_V1_loose.isPOGApproved  = cms.untracked.bool(True)
cutBasedElectronID_RunIIIWinter22_V1_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_RunIIIWinter22_V1_tight.isPOGApproved  = cms.untracked.bool(True)

