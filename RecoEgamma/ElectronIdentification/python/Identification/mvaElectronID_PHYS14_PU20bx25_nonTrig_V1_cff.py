from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

#
# The following MVA is derived for PHYS14 MC samples for non-triggering electrons.
# See more documentation in this presentation:
#     https://indico.cern.ch/event/367861/contribution/1/material/slides/0.pdf
#

# This MVA implementation class name
mvaPhys14NonTrigClassName = cms.string("ElectronMVAEstimatorRun2Phys14NonTrig")

# There are 6 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0   EB1 (eta<0.8)  pt 5-10 GeV
#   1   EB2 (eta>=0.8) pt 5-10 GeV
#   2   EE             pt 5-10 GeV
#   3   EB1 (eta<0.8)  pt 10-inf GeV
#   4   EB2 (eta>=0.8) pt 10-inf GeV
#   5   EE             pt 10-inf GeV

mvaPhys14NonTrigWeightFiles_V1 = cms.vstring(
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB1_5_oldscenario2phys14_BDT.weights.xml",
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB2_5_oldscenario2phys14_BDT.weights.xml",
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EE_5_oldscenario2phys14_BDT.weights.xml",
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB1_10_oldscenario2phys14_BDT.weights.xml",
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB2_10_oldscenario2phys14_BDT.weights.xml",
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EE_10_oldscenario2phys14_BDT.weights.xml"
    )

# Load some common definitions for MVA machinery
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *

# The locatoins of value maps with the actual MVA values and categories
# for all particles
mvaValueMapName      = "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Phys14NonTrigValues"
mvaCategoriesMapName = "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Phys14NonTrigCategories"

# The working point for this MVA that is expected to have about 80% signal
# efficiency on average separately for barrel and separately for endcap
# (averaged over pt 5-inf range)
idName = "mvaEleID-PHYS14-PU20bx25-nonTrig-V1-wp80"
MVA_WP80 = EleMVA_6Categories_WP(
    idName,
    mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = 0.738, # EB1 low pt
    cutCategory1 = 0.738, # EB2 low pt
    cutCategory2 = 0.730, # EE low pt
    cutCategory3 = 0.738, # EB1 
    cutCategory4 = 0.738, # EB2
    cutCategory5 = 0.730  # EE
    )

# The working point for this MVA that is expected to have about 90% signal
# efficiency on average separately for barrel and separately for endcap
# (averaged over pt 5-inf range)
idName = "mvaEleID-PHYS14-PU20bx25-nonTrig-V1-wp90"
MVA_WP90 = EleMVA_6Categories_WP(
    idName = idName,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = 0.479, # EB1 low pt
    cutCategory1 = 0.479, # EB2 low pt
    cutCategory2 = 0.479, # EE low pt 
    cutCategory3 = 0.479, # EB1       
    cutCategory4 = 0.479, # EB2       
    cutCategory5 = 0.479  # EE        
    )

#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaEleID_PHYS14_PU20bx25_nonTrig_V1_producer_config = cms.PSet( 
    mvaName            = mvaPhys14NonTrigClassName,
    mvaWeightFileNames = mvaPhys14NonTrigWeightFiles_V1
    )
# Create the VPset's for VID cuts
mvaEleID_PHYS14_PU20bx25_nonTrig_V1_wp80 = configureVIDMVAEleID_V1( MVA_WP80 )
mvaEleID_PHYS14_PU20bx25_nonTrig_V1_wp90 = configureVIDMVAEleID_V1( MVA_WP90 )

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register( mvaEleID_PHYS14_PU20bx25_nonTrig_V1_wp80.idName,
                              '293468bd0552c5b1a25c2dbbee4dcafc')
central_id_registry.register( mvaEleID_PHYS14_PU20bx25_nonTrig_V1_wp90.idName,
                              '7d2112062f92c4ea64d6430fc1fc10e0')
