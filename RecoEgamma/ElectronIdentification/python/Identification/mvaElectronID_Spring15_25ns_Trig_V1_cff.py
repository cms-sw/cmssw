from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *
import FWCore.ParameterSet.Config as cms

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

#
# The following MVA is derived for 25ns Spring15 MC samples for triggering electrons.
# See more documentation in this presentation (P.Pigard):
#     https://indico.cern.ch/event/369245/contribution/3/attachments/1153011/1655996/20150910_EID_POG_vAsPresented.pdf
#

# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "Spring15Trig25nsV1"

# There are 3 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0   EB1 (eta<0.8)  
#   1   EB2 (eta>=0.8) 
#   2   EE             

mvaSpring15TrigWeightFiles_V1 = cms.vstring(
    "RecoEgamma/ElectronIdentification/data/Spring15/EIDmva_EB1_10_oldTrigSpring15_25ns_data_1_VarD_TMVA412_Sig6BkgAll_MG_noSpec_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring15/EIDmva_EB2_10_oldTrigSpring15_25ns_data_1_VarD_TMVA412_Sig6BkgAll_MG_noSpec_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring15/EIDmva_EE_10_oldTrigSpring15_25ns_data_1_VarD_TMVA412_Sig6BkgAll_MG_noSpec_BDT.weights.xml.gz"
    )

# The working point for this MVA that is expected to have about 90% signal
# efficiency in each category
idName90 = "mvaEleID-Spring15-25ns-Trig-V1-wp90"
MVA_WP90 = EleMVA_WP(
    idName = idName90, mvaTag = mvaTag,
    cutCategory0 = "0.972153", # EB1 
    cutCategory1 = "0.922126", # EB2 
    cutCategory2 = "0.610764"  # EE 
    )

idName80 = "mvaEleID-Spring15-25ns-Trig-V1-wp80"
MVA_WP80 = EleMVA_WP(
    idName = idName80, mvaTag = mvaTag,
    cutCategory0 = "0.988153", # EB1 
    cutCategory1 = "0.96791" , # EB2 
    cutCategory2 = "0.841729"  # EE 
    )

#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaEleID_Spring15_25ns_Trig_V1_producer_config = cms.PSet( 
    mvaName            = cms.string(mvaClassName),
    mvaTag             = cms.string(mvaTag),
    # Category parameters
    nCategories         = cms.int32(3),
    categoryCuts        = EleMVA_3CategoriesCuts,
    # Weight files and variable definitions
    weightFileNames     = mvaSpring15TrigWeightFiles_V1,
    variableDefinition  = cms.string(mvaVariablesFile)
    )
# Create the VPset's for VID cuts
mvaEleID_Spring15_25ns_Trig_V1_wp90 = configureVIDMVAEleID( MVA_WP90 )
mvaEleID_Spring15_25ns_Trig_V1_wp80 = configureVIDMVAEleID( MVA_WP80 )

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(mvaEleID_Spring15_25ns_Trig_V1_wp90.idName,
                             'bb430b638bf3a4d970627021b0da63ae')
central_id_registry.register(mvaEleID_Spring15_25ns_Trig_V1_wp80.idName,
                             '81046ab478185af337be1be9b30948ae')

mvaEleID_Spring15_25ns_Trig_V1_wp90.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Spring15_25ns_Trig_V1_wp80.isPOGApproved = cms.untracked.bool(True)
