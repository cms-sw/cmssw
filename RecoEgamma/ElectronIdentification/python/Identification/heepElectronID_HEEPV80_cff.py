import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.heepElectronID_tools import HEEP_WorkingPoint_V1,configureHEEPElectronID_V80

#
# The HEEP ID cuts V8.0 below are high energy optimised cuts
# The cut values are taken from the twiki:
#       https://twiki.cern.ch/twiki/bin/view/CMS/HEEPElectronIdentificationRun2
# HEEP ID V8.0 is really V6.0 with a new name, the changes all happend to the object
# I bumped the name to make it was clear its for 9X

# The cut values for the  Barrel and Endcap
idName = "heepElectronID-HEEPV80"
WP_HEEP80_EB = HEEP_WorkingPoint_V1(
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

WP_HEEP80_EE = HEEP_WorkingPoint_V1(
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
heepElectronID_HEEPV80  = configureHEEPElectronID_V80 (idName, WP_HEEP80_EB, WP_HEEP80_EE )

#
# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(heepElectronID_HEEPV80.idName,"d997fbe84d9a0e48b959405def3bc7ec")
heepElectronID_HEEPV80.isPOGApproved = cms.untracked.bool(True)
 
