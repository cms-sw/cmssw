from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

#
# The following MVA is derived for 25ns Spring15 MC samples for non-triggering electrons.
# See more documentation in this presentation (P.Pigard):
#     https://indico.cern.ch/event/370506/contribution/1/attachments/1135340/1624370/20150726_EID_POG_circulating_vAsPresentedC.pdf
#

# This MVA implementation class name
mvaSpring15NonTrigClassName = "ElectronMVAEstimatorRun2Spring15NonTrig"
# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "25nsV1"

# There are 6 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0   EB1 (eta<0.8)  pt 5-10 GeV
#   1   EB2 (eta>=0.8) pt 5-10 GeV
#   2   EE             pt 5-10 GeV
#   3   EB1 (eta<0.8)  pt 10-inf GeV
#   4   EB2 (eta>=0.8) pt 10-inf GeV
#   5   EE             pt 10-inf GeV

mvaSpring15NonTrigWeightFiles_V1 = cms.vstring(
    "RecoEgamma/ElectronIdentification/data/Spring15/EIDmva_EB1_5_oldNonTrigSpring15_ConvVarCwoBoolean_TMVA412_FullStatLowPt_PairNegWeightsGlobal_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring15/EIDmva_EB2_5_oldNonTrigSpring15_ConvVarCwoBoolean_TMVA412_FullStatLowPt_PairNegWeightsGlobal_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring15/EIDmva_EE_5_oldNonTrigSpring15_ConvVarCwoBoolean_TMVA412_FullStatLowPt_PairNegWeightsGlobal_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring15/EIDmva_EB1_10_oldNonTrigSpring15_ConvVarCwoBoolean_TMVA412_FullStatLowPt_PairNegWeightsGlobal_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring15/EIDmva_EB2_10_oldNonTrigSpring15_ConvVarCwoBoolean_TMVA412_FullStatLowPt_PairNegWeightsGlobal_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring15/EIDmva_EE_10_oldNonTrigSpring15_ConvVarCwoBoolean_TMVA412_FullStatLowPt_PairNegWeightsGlobal_BDT.weights.xml.gz"
    )

# Load some common definitions for MVA machinery
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *

# The locatoins of value maps with the actual MVA values and categories
# for all particles.
# The names for the maps are "<module name>:<MVA class name>Values" 
# and "<module name>:<MVA class name>Categories"
mvaProducerModuleLabel = "electronMVAValueMapProducer"
mvaValueMapName        = mvaProducerModuleLabel + ":" + mvaSpring15NonTrigClassName + mvaTag + "Values"
mvaCategoriesMapName   = mvaProducerModuleLabel + ":" + mvaSpring15NonTrigClassName + mvaTag + "Categories"

# The working point for this MVA that is expected to have about 90% signal
# efficiency in each category
idName90 = "mvaEleID-Spring15-25ns-nonTrig-V1-wp90"
MVA_WP90 = EleMVA_6Categories_WP(
    idName = idName90,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = -0.083313, # EB1 low pt
    cutCategory1 = -0.235222, # EB2 low pt
    cutCategory2 = -0.67099, # EE low pt 
    cutCategory3 =  0.913286, # EB1       
    cutCategory4 =  0.805013, # EB2       
    cutCategory5 =  0.358969  # EE        
    )

idName80 = "mvaEleID-Spring15-25ns-nonTrig-V1-wp80"
MVA_WP80 = EleMVA_6Categories_WP(
    idName = idName80,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 =  0.287435, # EB1 low pt
    cutCategory1 =  0.221846, # EB2 low pt
    cutCategory2 = -0.303263, # EE low pt 
    cutCategory3 =  0.967083, # EB1       
    cutCategory4 =  0.929117, # EB2       
    cutCategory5 =  0.726311  # EE        
    )

### WP tuned for HZZ analysis with very high efficiency (about 98%)
idNameLoose = "mvaEleID-Spring15-25ns-nonTrig-V1-wpLoose"
MVA_WPLoose = EleMVA_6Categories_WP(
    idName = idNameLoose,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 =  -0.265, # EB1 low pt
    cutCategory1 =  -0.556, # EB2 low pt
    cutCategory2 =  -0.551, # EE low pt
    cutCategory3 =  -0.072, # EB1
    cutCategory4 =  -0.286, # EB2
    cutCategory5 =  -0.267  # EE
    )


#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaEleID_Spring15_25ns_nonTrig_V1_producer_config = cms.PSet( 
    mvaName            = cms.string(mvaSpring15NonTrigClassName),
    mvaTag             = cms.string(mvaTag),
    # This MVA uses conversion info, so configure several data items on that
    beamSpot           = cms.InputTag('offlineBeamSpot'),
    conversionsAOD     = cms.InputTag('allConversions'),
    conversionsMiniAOD = cms.InputTag('reducedEgamma:reducedConversions'),
    #
    weightFileNames    = mvaSpring15NonTrigWeightFiles_V1
    )
# Create the VPset's for VID cuts
mvaEleID_Spring15_25ns_nonTrig_V1_wpLoose = configureVIDMVAEleID_V1( MVA_WPLoose )
mvaEleID_Spring15_25ns_nonTrig_V1_wp90    = configureVIDMVAEleID_V1( MVA_WP90 )
mvaEleID_Spring15_25ns_nonTrig_V1_wp80    = configureVIDMVAEleID_V1( MVA_WP80 )


# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(mvaEleID_Spring15_25ns_nonTrig_V1_wpLoose.idName,
                             '99ff36834c4342110d84ea2350a1229c')
central_id_registry.register(mvaEleID_Spring15_25ns_nonTrig_V1_wp90.idName,
                             'ac4fdc160eefe9eae7338601c02ed4bb')
central_id_registry.register(mvaEleID_Spring15_25ns_nonTrig_V1_wp80.idName,
                             '113c47ceaea0fa687b8bd6d880eb4957')

mvaEleID_Spring15_25ns_nonTrig_V1_wpLoose.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Spring15_25ns_nonTrig_V1_wp90.isPOGApproved    = cms.untracked.bool(True)
mvaEleID_Spring15_25ns_nonTrig_V1_wp80.isPOGApproved    = cms.untracked.bool(True)
