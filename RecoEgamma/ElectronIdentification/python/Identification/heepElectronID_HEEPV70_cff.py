import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.heepElectronID_tools import HEEP_WorkingPoint_V1,configureHEEPElectronID_V70

#
# The HEEP ID cuts V6.0 below are optimized IDs for PHYS14 Scenario PU 20, bx 25ns
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/view/CMS/HEEPElectronIdentificationRun2#Selection_Cuts_HEEP_V5_1
#       (where they may not stay, if a newer version of cuts becomes available for these
#        conditions)
# See also the presentation explaining these working points (this will not change):
#   [ ... none available for this particular version ... ]


# The cut values for the  Barrel and Endcap
idName = "heepElectronID-HEEPV70"
WP_HEEP70_EB = HEEP_WorkingPoint_V1(
    idName=idName,     
    dEtaInSeedCut=0.004,     
    dPhiInCut=0.06,      
    full5x5SigmaIEtaIEtaCut=9999,     
    # Two constants for the GsfEleFull5x5E2x5OverE5x5Cut
    minE1x5OverE5x5Cut=0.83,    
    minE2x5OverE5x5Cut=0.94,     
    # Three constants for the GsfEleHadronicOverEMLinearCut
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    hOverESlopeTerm=0.05,  
    hOverESlopeStart=0.00,    
    hOverEConstTerm=1.00,    
    # Three constants for the GsfEleTrkPtIsoCut: 
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    trkIsoSlopeTerm=0.00,     
    trkIsoSlopeStart=0.00,   
    trkIsoConstTerm=5.00,     
    # Three constants for the GsfEleEmHadD1IsoRhoCut: 
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    # Also for the same cut, the effective area for the rho correction of the isolation
    ehIsoSlopeTerm=0.03,       
    ehIsoSlopeStart=0.00,       
    ehIsoConstTerm=2.00,        
    effAreaForEHIso=0.28,        
    # other cuts
    dxyCut=0.02,
    maxMissingHitsCut=1,
    ecalDrivenCut=1
    )

WP_HEEP70_EE = HEEP_WorkingPoint_V1(
     idName=idName,     
    dEtaInSeedCut=0.006,     
    dPhiInCut=0.06,      
    full5x5SigmaIEtaIEtaCut=0.03,     
    # Two constants for the GsfEleFull5x5E2x5OverE5x5Cut
    minE1x5OverE5x5Cut=-1.0,    
    minE2x5OverE5x5Cut=-1.0,     
    # Three constants for the GsfEleHadronicOverEMLinearCut
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    hOverESlopeTerm=0.05,  
    hOverESlopeStart=0.00,    
    hOverEConstTerm=5,    
    # Three constants for the GsfEleTrkPtIsoCut: 
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    trkIsoSlopeTerm=0.00,     
    trkIsoSlopeStart=0.00,   
    trkIsoConstTerm=5.00,     
    # Three constants for the GsfEleEmHadD1IsoRhoCut: 
    #     cut = constTerm if value < slopeStart
    #     cut = slopeTerm * (value - slopeStart) + constTerm if value >= slopeStart
    # Also for the same cut, the effective area for the rho correction of the isolation
    ehIsoSlopeTerm=0.03,       
    ehIsoSlopeStart=50.0,       
    ehIsoConstTerm=2.50,        
    effAreaForEHIso=0.28,        
    # other cuts
    dxyCut=0.05,
    maxMissingHitsCut=1,
    ecalDrivenCut=1
  
    )

#
# Finally, set up VID configuration for all cuts
#
heepElectronID_HEEPV70  = configureHEEPElectronID_V70 (idName, WP_HEEP70_EB, WP_HEEP70_EE )

#
# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(heepElectronID_HEEPV70.idName,"45d49ea5f46f3f13f579d208e5e3412b")
heepElectronID_HEEPV70.isPOGApproved = cms.untracked.bool(True)
 
