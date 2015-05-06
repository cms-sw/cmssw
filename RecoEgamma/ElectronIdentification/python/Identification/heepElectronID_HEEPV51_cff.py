from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.heepElectronID_tools import *

#
# The HEEP ID cuts V5.1 below are optimized IDs for PHYS14 Scenario PU 20, bx 25ns
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/view/CMS/HEEPElectronIdentificationRun2#Selection_Cuts_HEEP_V5_1
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
#   [ ... none available for this particular version ... ]


# The cut values for the  Barrel and Endcap
idName = "heepElectronID-HEEPV51"
WP_HEEP51_EB = HEEP_WorkingPoint_V1(
    idName    ,  # idName
    0.004     ,  # dEtaInSeedCut,
    0.06      ,  # dPhiInCut,
    9999      ,  # full5x5SigmaIEtaIEtaCut,
    # Two constants for the GsfEleFull5x5E2x5OverE5x5Cut
    0.83      ,  # minE1x5OverE5x5Cut,
    0.94      ,  # minE2x5OverE5x5Cut,
    # Three constants for the GsfEleHadronicOverEMLinearCut
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    0.05      ,  # hOverESlopeTerm,
    0.00      ,  # hOverESlopeStart,
    2.00      ,  # hOverEConstTerm,
    # Three constants for the GsfEleTrkPtIsoCut: 
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    0.00      ,  # trkIsoSlopeTerm,
    0.00      ,  # trkIsoSlopeStart,
    5.00      ,  # trkIsoConstTerm,
    # Three constants for the GsfEleEmHadD1IsoRhoCut: 
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    # Also for the same cut, the effective area for the rho correction of the isolation
    0.03      ,  # ehIsoSlopeTerm,
    0.00      ,  # ehIsoSlopeStart,
    2.00      ,  # ehIsoConstTerm,
    0.28      ,  # effAreaForEHIso, presently same for barrel and endcap
    # other cuts
    0.02      ,  # dxyCut,
    1            # maxMissingHitsCut
    )

WP_HEEP51_EE = HEEP_WorkingPoint_V1(
    idName    ,  # idName
    0.006     ,  # dEtaInSeedCut,
    0.06      ,  # dPhiInCut,
    0.03      ,  # full5x5SigmaIEtaIEtaCut,
    # Two constants for the GsfEleFull5x5E2x5OverE5x5Cut
    -1.0      ,  # minE1x5OverE5x5Cut,
    -1.0      ,  # minE2x5OverE5x5Cut,
    # Three constants for the GsfEleHadronicOverEMLinearCut
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    0.05      ,  # hOverESlopeTerm,
    0.00      ,  # hOverESlopeStart,
    12.5      ,  # hOverEConstTerm,
    # Three constants for the GsfEleTrkPtIsoCut: 
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    0.00      ,  # trkIsoSlopeTerm,
    0.00      ,  # trkIsoSlopeStart,
    5.00      ,  # trkIsoConstTerm,
    # Three constants for the GsfEleEmHadD1IsoRhoCut: 
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    # Also for the same cut, the effective area for the rho correction of the isolation
    0.03      ,  # ehIsoSlopeTerm,
    50.0      ,  # ehIsoSlopeStart,
    2.50      ,  # ehIsoConstTerm,
    0.28      ,  # effAreaForEHIso, presently same for barrel and endcap
    # other cuts
    0.05      ,  # dxyCut,
    1            # maxMissingHitsCut
    )

#
# Finally, set up VID configuration for all cuts
#
heepElectronID_HEEPV51  = configureHEEPElectronID_V51 ( WP_HEEP51_EB, WP_HEEP51_EE )

#
# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(heepElectronID_HEEPV51.idName,"d930c34a554f86e6d4e8c247c4737140")
