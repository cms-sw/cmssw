from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_tools \
    import ( EleWorkingPoint_V3,
             configureVIDCutBasedEleID_V3 )

#
# The ID cuts below are optimized IDs on Spring16 simulation with 80X-based production
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
#        https://indico.cern.ch/event/482677/contributions/2259342/attachments/1316731/1972911/talk_electron_ID_spring16_update.pdf
#
# First, define cut values
#

# Veto working point Barrel and Endcap
idName = "cutBasedElectronID-Summer16-80X-V1-veto"
WP_Veto_EB = EleWorkingPoint_V3(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0115  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00749 , # dEtaInSeedCut
    dPhiInCut                      = 0.228   , # dPhiInCut
    hOverECut                      = 0.356   , # hOverECut
    relCombIsolationWithEALowPtCut = 0.175   , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.175   , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.299   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 2          # missingHitsCut
    )

WP_Veto_EE = EleWorkingPoint_V3(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0370  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00895 , # dEtaInSeedCut
    dPhiInCut                      = 0.213   , # dPhiInCut
    hOverECut                      = 0.211   , # hOverECut
    relCombIsolationWithEALowPtCut = 0.159   , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.159   , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.150   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 3          # missingHitsCut
    )

# Loose working point Barrel and Endcap
idName = "cutBasedElectronID-Summer16-80X-V1-loose"
WP_Loose_EB = EleWorkingPoint_V3(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0110  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00477 , # dEtaInSeedCut
    dPhiInCut                      = 0.222   , # dPhiInCut
    hOverECut                      = 0.298   , # hOverECut
    relCombIsolationWithEALowPtCut = 0.0994  , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.0994  , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.241   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Loose_EE = EleWorkingPoint_V3(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0314  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00868 , # dEtaInSeedCut
    dPhiInCut                      = 0.213   , # dPhiInCut
    hOverECut                      = 0.101   , # hOverECut
    relCombIsolationWithEALowPtCut = 0.107   , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.107   , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.140   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1         # missingHitsCut
    )

# Medium working point Barrel and Endcap
idName = "cutBasedElectronID-Summer16-80X-V1-medium"
WP_Medium_EB = EleWorkingPoint_V3(
    idName                         = idName , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.00998, # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00311, # dEtaInSeedCut
    dPhiInCut                      = 0.103  , # dPhiInCut
    hOverECut                      = 0.253  , # hOverECut
    relCombIsolationWithEALowPtCut = 0.0695 , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.0695 , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.134  , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Medium_EE = EleWorkingPoint_V3(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0298  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00609 , # dEtaInSeedCut
    dPhiInCut                      = 0.0450  , # dPhiInCut
    hOverECut                      = 0.0878  , # hOverECut
    relCombIsolationWithEALowPtCut = 0.0821  , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.0821  , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.130   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

# Tight working point Barrel and Endcap
idName = "cutBasedElectronID-Summer16-80X-V1-tight"
WP_Tight_EB = EleWorkingPoint_V3(
    idName                         = idName    , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.00998   , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00308   , # dEtaInSeedCut
    dPhiInCut                      = 0.0816    , # dPhiInCut
    hOverECut                      = 0.0414    , # hOverECut
    relCombIsolationWithEALowPtCut = 0.0588    , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.0588    , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.0129    , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Tight_EE = EleWorkingPoint_V3(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0292  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00605 , # dEtaInSeedCut
    dPhiInCut                      = 0.0394  , # dPhiInCut
    hOverECut                      = 0.0641  , # hOverECut
    relCombIsolationWithEALowPtCut = 0.0571  , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.0571  , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.0129 , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

# Second, define what effective areas to use for pile-up correction
isoEffAreas = "RecoEgamma/ElectronIdentification/data/Summer16/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_80X.txt"


#
# Set up VID configuration for all cuts and working points
#

cutBasedElectronID_Summer16_80X_V1_veto = configureVIDCutBasedEleID_V3(WP_Veto_EB, WP_Veto_EE, isoEffAreas)
cutBasedElectronID_Summer16_80X_V1_loose = configureVIDCutBasedEleID_V3(WP_Loose_EB, WP_Loose_EE, isoEffAreas)
cutBasedElectronID_Summer16_80X_V1_medium = configureVIDCutBasedEleID_V3(WP_Medium_EB, WP_Medium_EE, isoEffAreas)
cutBasedElectronID_Summer16_80X_V1_tight = configureVIDCutBasedEleID_V3(WP_Tight_EB, WP_Tight_EE, isoEffAreas)


# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedElectronID_Summer16_80X_V1_veto.idName,
                             '0025c1841da1ab64a08d703ded72409b')
central_id_registry.register(cutBasedElectronID_Summer16_80X_V1_loose.idName,
                             'c1c4c739f1ba0791d40168c123183475')
central_id_registry.register(cutBasedElectronID_Summer16_80X_V1_medium.idName,
                             '71b43f74a27d2fd3d27416afd22e8692')
central_id_registry.register(cutBasedElectronID_Summer16_80X_V1_tight.idName,
                             'ca2a9db2976d80ba2c13f9bfccdc32f2')


### for now until we have a database...
cutBasedElectronID_Summer16_80X_V1_veto.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_Summer16_80X_V1_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_Summer16_80X_V1_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_Summer16_80X_V1_tight.isPOGApproved = cms.untracked.bool(True)
