from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_tools \
    import ( EleWorkingPoint_V5,
             configureVIDCutBasedEleID_V5 )

#############################This ID is for Phase II and hence matters for EB
#
# The ID cuts below are optimized IDs on Summer20 Phase II simulation 
# These are very initial tuning which will improve further as we improve 
###our understanding on the variables for phase II (e.g noise cleaned sieie is not yet
###used here. PF cluster isolations may be better but not yet used). We just use the Run II variables on phase II samples for the tuning 
# See also the presentation explaining these working points:
#  https://indico.cern.ch/event/1000891/contributions/4203637/attachments/2183448/3688820/ElectronIDtunning_EgammaMeeting_03Feb2021.pdf
# https://indico.cern.ch/event/879937/contributions/4108369/attachments/2147436/3619904/EgammaID_phaseIIElectrontunning_pyu_2020Nov20.pdf
#
#

# Veto working point Barrel
idName = "cutBasedElectronID-Summer20-PhaseII-V0-veto"
WP_Veto_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       =  0.0181 , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  =  0.00548, # dEtaInSeedCut
    dPhiInCut                      =  0.197, # dPhiInCut
    hOverECut_C0                   =  0.313   , # hOverECut
    hOverECut_CE                   =  0.   ,
    hOverECut_Cr                   =  0.  ,
    relCombIsolationWithEACut_C0   =  0.284  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  =  0.  ,
    absEInverseMinusPInverseCut    =  0.203  , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 =  2         # missingHitsCut
    )

WP_Veto_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       =  0.0181 , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  =  0.00548, # dEtaInSeedCut
    dPhiInCut                      =  0.197, # dPhiInCut
    hOverECut_C0                   =  0.313   , # hOverECut
    hOverECut_CE                   =  0.   ,
    hOverECut_Cr                   =  0.  ,
    relCombIsolationWithEACut_C0   =  0.284  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  =  0.  ,
    absEInverseMinusPInverseCut    =  0.203  , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 =  2         # missingHitsCut
    )

# Loose working point Barrel and Endcap
idName = "cutBasedElectronID-Summer20-PhaseII-V0-loose"
WP_Loose_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0162  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00409 , # dEtaInSeedCut
    dPhiInCut                      = 0.0679  , # dPhiInCut
    hOverECut_C0                   = 0.222    , # hOverECut
    hOverECut_CE                   = 0.    ,
    hOverECut_Cr                   = 0.  ,
    relCombIsolationWithEACut_C0   = 0.223   , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.   ,
    absEInverseMinusPInverseCut    = 0.0747   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Loose_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0162  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00409 , # dEtaInSeedCut
    dPhiInCut                      = 0.0679  , # dPhiInCut
    hOverECut_C0                   = 0.222    , # hOverECut
    hOverECut_CE                   = 0.    ,
    hOverECut_Cr                   = 0.  ,
    relCombIsolationWithEACut_C0   = 0.223   , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.   ,
    absEInverseMinusPInverseCut    = 0.0747   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

# Medium working point Barrel and Endcap
idName = "cutBasedElectronID-Summer20-PhaseII-V0-medium"
WP_Medium_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0156 , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00326  , # dEtaInSeedCut
    dPhiInCut                      = 0.0434  , # dPhiInCut
    hOverECut_C0                   = 0.138   , # hOverECut
    hOverECut_CE                   = 0.    ,
    hOverECut_Cr                   = 0.  ,
    relCombIsolationWithEACut_C0   = 0.159  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.   ,
    absEInverseMinusPInverseCut    = 0.0735   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Medium_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0156 , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00326  , # dEtaInSeedCut
    dPhiInCut                      = 0.0434  , # dPhiInCut
    hOverECut_C0                   = 0.138   , # hOverECut
    hOverECut_CE                   = 0.    ,
    hOverECut_Cr                   = 0.  ,
    relCombIsolationWithEACut_C0   = 0.159  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.   ,
    absEInverseMinusPInverseCut    = 0.0735   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )


# Tight working point Barrel and Endcap
idName = "cutBasedElectronID-Summer20-PhaseII-V0-tight"
WP_Tight_EB = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0137  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00325 , # dEtaInSeedCut
    dPhiInCut                      = 0.0365   , # dPhiInCut
    hOverECut_C0                   = 0.103   , # hOverECut
    hOverECut_CE                   = 0.    ,
    hOverECut_Cr                   = 0.  ,
    relCombIsolationWithEACut_C0   = 0.121  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.   ,
    absEInverseMinusPInverseCut    = 0.0161   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )

WP_Tight_EE = EleWorkingPoint_V5(
    idName                         = idName  , # idName
    full5x5_sigmaIEtaIEtaCut       = 0.0137  , # full5x5_sigmaIEtaIEtaCut
    dEtaInSeedCut                  = 0.00325 , # dEtaInSeedCut
    dPhiInCut                      = 0.0365   , # dPhiInCut
    hOverECut_C0                   = 0.103   , # hOverECut
    hOverECut_CE                   = 0.    ,
    hOverECut_Cr                   = 0.  ,
    relCombIsolationWithEACut_C0   = 0.121  , # relCombIsolationWithEACut
    relCombIsolationWithEACut_Cpt  = 0.   ,
    absEInverseMinusPInverseCut    = 0.0161   , # absEInverseMinusPInverseCut
    # conversion veto cut needs no parameters, so not mentioned
    missingHitsCut                 = 1          # missingHitsCut
    )
# Second, define what effective areas to use for pile-up correction
isoEffAreas = "RecoEgamma/ElectronIdentification/data/PhaseII/EffectiveArea_electrons_barrel_PhaseII.txt"

#
# Set up VID configuration for all cuts and working points
#

cutBasedElectronID_Summer20_PhaseII_V0_veto   = configureVIDCutBasedEleID_V5(WP_Veto_EB,   WP_Veto_EE, isoEffAreas)
cutBasedElectronID_Summer20_PhaseII_V0_loose  = configureVIDCutBasedEleID_V5(WP_Loose_EB,  WP_Loose_EE, isoEffAreas)
cutBasedElectronID_Summer20_PhaseII_V0_medium = configureVIDCutBasedEleID_V5(WP_Medium_EB, WP_Medium_EE, isoEffAreas)
cutBasedElectronID_Summer20_PhaseII_V0_tight  = configureVIDCutBasedEleID_V5(WP_Tight_EB,  WP_Tight_EE, isoEffAreas)

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry and the isPOGApproved lines,
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(cutBasedElectronID_Summer20_PhaseII_V0_veto.idName,   '75ed578c1b442ee97b1a2e50263e1aa3')
central_id_registry.register(cutBasedElectronID_Summer20_PhaseII_V0_loose.idName,  'c43597fa43676ea1f444d06c701866dc')
central_id_registry.register(cutBasedElectronID_Summer20_PhaseII_V0_medium.idName, '1ea055b2139ad578dbfbdec1f7c78715')
central_id_registry.register(cutBasedElectronID_Summer20_PhaseII_V0_tight.idName,  '3c85a35b6dfbf25713db70876cb6675b')

### for now until we have a database...
cutBasedElectronID_Summer20_PhaseII_V0_veto.isPOGApproved   = cms.bool(False)
cutBasedElectronID_Summer20_PhaseII_V0_loose.isPOGApproved  = cms.bool(False)
cutBasedElectronID_Summer20_PhaseII_V0_medium.isPOGApproved = cms.bool(False)
cutBasedElectronID_Summer20_PhaseII_V0_tight.isPOGApproved  = cms.bool(False)
