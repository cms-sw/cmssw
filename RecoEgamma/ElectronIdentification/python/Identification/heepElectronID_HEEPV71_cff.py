import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.heepElectronID_tools import HEEP_WorkingPoint_V1,configureHEEPElectronID_V71

#the same HEEP V70 but now has calo et cut of >5 GeV and isolation cut is relaxed vs calo et

# The cut values for the  Barrel and Endcap
idName = "heepElectronID-HEEPV71"
WP_HEEP71_EB = HEEP_WorkingPoint_V1(
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

WP_HEEP71_EE = HEEP_WorkingPoint_V1(
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
heepElectronID_HEEPV71  = configureHEEPElectronID_V71 (idName, WP_HEEP71_EB, WP_HEEP71_EE, 5. )

#
# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

#central_id_registry.register(heepElectronID_HEEPV71.idName,"49b6b60e9f16727f241eb34b9d345a8f")
heepElectronID_HEEPV71.isPOGApproved = cms.untracked.bool(True)
 
