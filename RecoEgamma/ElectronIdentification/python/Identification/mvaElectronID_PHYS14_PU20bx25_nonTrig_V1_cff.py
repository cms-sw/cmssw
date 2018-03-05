from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *
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

# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "Phys14NonTrig25nsV1"

# There are 6 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0   EB1 (eta<0.8)  pt 5-10 GeV
#   1   EB2 (eta>=0.8) pt 5-10 GeV
#   2   EE             pt 5-10 GeV
#   3   EB1 (eta<0.8)  pt 10-inf GeV
#   4   EB2 (eta>=0.8) pt 10-inf GeV
#   5   EE             pt 10-inf GeV

mvaPhys14NonTrigWeightFiles_V1 = cms.vstring(
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB1_5_oldscenario2phys14_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB2_5_oldscenario2phys14_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EE_5_oldscenario2phys14_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB1_10_oldscenario2phys14_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EB2_10_oldscenario2phys14_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/PHYS14/EIDmva_EE_10_oldscenario2phys14_BDT.weights.xml.gz"
    )

# The working point for this MVA that is expected to have about 80% signal
# efficiency on in each category
idName = "mvaEleID-PHYS14-PU20bx25-nonTrig-V1-wp80"
MVA_WP80 = EleMVA_WP(
    idName, mvaTag,
    cutCategory0 = "-0.253", # EB1 low pt
    cutCategory1 = " 0.081", # EB2 low pt
    cutCategory2 = "-0.081", # EE low pt
    cutCategory3 = " 0.965", # EB1 
    cutCategory4 = " 0.917", # EB2
    cutCategory5 = " 0.683"  # EE
    )

# The working point for this MVA that is expected to have about 90% signal
# efficiency in each category
idName = "mvaEleID-PHYS14-PU20bx25-nonTrig-V1-wp90"
MVA_WP90 = EleMVA_WP(
    idName, mvaTag,
    cutCategory0 = "-0.483", # EB1 low pt
    cutCategory1 = "-0.267", # EB2 low pt
    cutCategory2 = "-0.323", # EE low pt 
    cutCategory3 = " 0.933", # EB1       
    cutCategory4 = " 0.825", # EB2       
    cutCategory5 = " 0.337"  # EE        
    )

#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaEleID_PHYS14_PU20bx25_nonTrig_V1_producer_config = cms.PSet( 
    mvaName            = cms.string(mvaClassName),
    mvaTag             = cms.string(mvaTag),
    # Category parameters
    nCategories        = cms.int32(6),
    categoryCuts       = EleMVA_6CategoriesCuts,
    # Weight files and variable definitions
    weightFileNames    = mvaPhys14NonTrigWeightFiles_V1,
    variableDefinition = cms.string(mvaVariablesFileClassic)
    )
# Create the VPset's for VID cuts
mvaEleID_PHYS14_PU20bx25_nonTrig_V1_wp80 = configureVIDMVAEleID( MVA_WP80 )
mvaEleID_PHYS14_PU20bx25_nonTrig_V1_wp90 = configureVIDMVAEleID( MVA_WP90 )

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to
# 1) comment out the lines below about the registry,
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register( mvaEleID_PHYS14_PU20bx25_nonTrig_V1_wp80.idName,
                              '768465d41956da069c83bf245398d5e6')
central_id_registry.register( mvaEleID_PHYS14_PU20bx25_nonTrig_V1_wp90.idName,
                              '7d091368510c32f0ab29a53323cae95a')

mvaEleID_PHYS14_PU20bx25_nonTrig_V1_wp80.isPOGApproved = cms.untracked.bool(False)
mvaEleID_PHYS14_PU20bx25_nonTrig_V1_wp90.isPOGApproved = cms.untracked.bool(False)
