import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Common functions and classes for ID definition are imported here:
from RecoEgamma.ElectronIdentification.Identification.heepElectronID_tools import HEEP_WorkingPoint_V1,configureHEEPElectronID_V71
from RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff import WP_HEEP70_EB,WP_HEEP70_EE
import copy
#the same HEEP V70 but now has calo et cut of >5 GeV and isolation cut is relaxed vs calo et

# The cut values for the  Barrel and Endcap
idName = "heepElectronID-HEEPV71"
WP_HEEP71_EB = copy.deepcopy(WP_HEEP70_EB)
WP_HEEP71_EB.idName = str(idName)
WP_HEEP71_EE = copy.deepcopy(WP_HEEP70_EE)
WP_HEEP71_EE.idName = str(idName)
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
 
