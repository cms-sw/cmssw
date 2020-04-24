from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_tools \
    import ( EleWorkingPoint_V4,
             IsolationCutInputs_V2,
             configureVIDCutBasedEleID_V4 )

#
# The ID cuts below are optimized IDs on Spring16 simulation with 80X-based production
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
# https://indico.cern.ch/event/662751/contributions/2778044/attachments/1562080/2459801/171121_egamma_workshop.pdf
#
# First, define cut values
#

# Veto working point Barrel and Endcap
#V1 of IDs good for Moriond 18
idName = "cutBasedElectronID-Fall17-94X-V1-veto"
WP_Veto_EB = EleWorkingPoint_V4(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0128  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00523 , # dEtaInSeedCut
    dPhiInCut                      = 0.159   , # dPhiInCut
    hOverECut_C0                   = 0.05    , # hOverECut
    hOverECut_CE                   = 1.12    ,
    hOverECut_Cr                   = 0.0368  ,
    relCombIsolationWithEALowPtCut = 0.168   , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.168   , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.193   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 2          # missingHitsCut
    )

WP_Veto_EE = EleWorkingPoint_V4(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0445  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00984 , # dEtaInSeedCut
    dPhiInCut                      = 0.157   , # dPhiInCut
    hOverECut_C0                   = 0.05    , # hOverECut
    hOverECut_CE                   = 0.5     ,
    hOverECut_Cr                   = 0.201   ,
    relCombIsolationWithEALowPtCut = 0.185   , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.185   , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.0962   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 3          # missingHitsCut
    )

# Loose working point Barrel and Endcap
idName = "cutBasedElectronID-Fall17-94X-V1-loose"
WP_Loose_EB = EleWorkingPoint_V4(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0105  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00387 , # dEtaInSeedCut
    dPhiInCut                      = 0.0716   , # dPhiInCut
    hOverECut_C0                   = 0.05    , # hOverECut
    hOverECut_CE                   = 1.12    ,
    hOverECut_Cr                   = 0.0368  ,
    relCombIsolationWithEALowPtCut = 0.133  , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.133  , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.129   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Loose_EE = EleWorkingPoint_V4(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0356  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.0072 , # dEtaInSeedCut
    dPhiInCut                      = 0.147   , # dPhiInCut
    hOverECut_C0                   = 0.0414  , # hOverECut
    hOverECut_CE                   = 0.5     ,
    hOverECut_Cr                   = 0.201   ,
    relCombIsolationWithEALowPtCut = 0.146   , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.146   , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.0875   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1         # missingHitsCut
    )

# Medium working point Barrel and Endcap
idName = "cutBasedElectronID-Fall17-94X-V1-medium"
WP_Medium_EB = EleWorkingPoint_V4(
    idName                         = idName , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0105, # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00365, # dEtaInSeedCut
    dPhiInCut                      = 0.0588  , # dPhiInCut
    hOverECut_C0                   = 0.026   , # hOverECut
    hOverECut_CE                   = 1.12    ,
    hOverECut_Cr                   = 0.0368  ,
    relCombIsolationWithEALowPtCut = 0.0718 , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.0718 , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.0327  , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Medium_EE = EleWorkingPoint_V4(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0309  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00625 , # dEtaInSeedCut
    dPhiInCut                      = 0.0355  , # dPhiInCut
    hOverECut_C0                   = 0.026    , # hOverECut
    hOverECut_CE                   = 0.5     ,
    hOverECut_Cr                   = 0.201   ,
    relCombIsolationWithEALowPtCut = 0.143  , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.143  , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.0335   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

# Tight working point Barrel and Endcap
idName = "cutBasedElectronID-Fall17-94X-V1-tight"
WP_Tight_EB = EleWorkingPoint_V4(
    idName                         = idName    , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0104   , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00353   , # dEtaInSeedCut
    dPhiInCut                      = 0.0499    , # dPhiInCut
    hOverECut_C0                   = 0.026   , # hOverECut
    hOverECut_CE                   = 1.12    ,
    hOverECut_Cr                   = 0.0368  ,
    relCombIsolationWithEALowPtCut = 0.0361    , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.0361    , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.0278    , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Tight_EE = EleWorkingPoint_V4(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0305  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00567 , # dEtaInSeedCut
    dPhiInCut                      = 0.0165  , # dPhiInCut
    hOverECut_C0                   = 0.026   , # hOverECut
    hOverECut_CE                   = 0.5     ,
    hOverECut_Cr                   = 0.201   ,
    relCombIsolationWithEALowPtCut = 0.094  , # relCombIsolationWithEALowPtCut
    relCombIsolationWithEAHighPtCut= 0.094  , # relCombIsolationWithEAHighPtCut
    absEInverseMinusPInverseCut    = 0.0158 , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

# Second, define what effective areas to use for pile-up correction
isoInputs = IsolationCutInputs_V2(
    # phoIsolationEffAreas
    "RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_92X.txt"
)


#
# Set up VID configuration for all cuts and working points
#

cutBasedElectronID_Fall17_94X_V1_veto = configureVIDCutBasedEleID_V4(WP_Veto_EB, WP_Veto_EE, isoInputs)
cutBasedElectronID_Fall17_94X_V1_loose = configureVIDCutBasedEleID_V4(WP_Loose_EB, WP_Loose_EE, isoInputs)
cutBasedElectronID_Fall17_94X_V1_medium = configureVIDCutBasedEleID_V4(WP_Medium_EB, WP_Medium_EE, isoInputs)
cutBasedElectronID_Fall17_94X_V1_tight = configureVIDCutBasedEleID_V4(WP_Tight_EB, WP_Tight_EE, isoInputs)


# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedElectronID_Fall17_94X_V1_veto.idName,
                             '43be9b381a8d9b0910b7f81a5ad8ff3a')
central_id_registry.register(cutBasedElectronID_Fall17_94X_V1_loose.idName,
                             '0b8456d622494441fe713a6858e0f7c1')
central_id_registry.register(cutBasedElectronID_Fall17_94X_V1_medium.idName,
                             'a238ee70910de53d36866e89768500e9')
central_id_registry.register(cutBasedElectronID_Fall17_94X_V1_tight.idName,
                             '4acb2d2796efde7fba75380ce8823fc2')


### for now until we have a database...
cutBasedElectronID_Fall17_94X_V1_veto.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_Fall17_94X_V1_loose.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_Fall17_94X_V1_medium.isPOGApproved = cms.untracked.bool(True)
cutBasedElectronID_Fall17_94X_V1_tight.isPOGApproved = cms.untracked.bool(True)
