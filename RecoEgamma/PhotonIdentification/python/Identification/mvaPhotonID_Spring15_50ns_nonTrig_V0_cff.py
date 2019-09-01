from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_tools import *

mvaVariablesFile        = "RecoEgamma/PhotonIdentification/data/PhotonMVAEstimatorRun2VariablesSpring15ValMaps.txt"

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

#
# The following MVA is derived for Spring15 MC samples for non-triggering photons.
# See more documentation in this presentation:
#    https://indico.cern.ch/event/369233/contribution/3/material/slides/0.pdf
#

# This MVA implementation class name
mvaSpring15NonTrigClassName = "PhotonMVAEstimator"
# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "Run2Spring15NonTrig50nsV0"

# There are 2 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0    barrel photons
#   1    endcap photons

mvaSpring15NonTrigWeightFiles_V0 = cms.vstring(
    path.join(weightFileBaseDir, "Spring15/50ns_EB_V0.weights.xml.gz"),
    path.join(weightFileBaseDir, "Spring15/50ns_EE_V0.weights.xml.gz"),
    )

# The locatoins of value maps with the actual MVA values and categories
# for all particles.
# The names for the maps are "<module name>:<MVA class name>Values"
# and "<module name>:<MVA class name>Categories"
mvaProducerModuleLabel = "photonMVAValueMapProducer"
mvaValueMapName        = mvaProducerModuleLabel + ":" + mvaSpring15NonTrigClassName + mvaTag + "Values"
mvaCategoriesMapName   = mvaProducerModuleLabel + ":" + mvaSpring15NonTrigClassName + mvaTag + "Categories"

# The working point for this MVA that is expected to have about 90% signal
# efficiency in each category for photons with pt>30 GeV (somewhat lower
# for lower pt photons).
idName = "mvaPhoID-Spring15-50ns-nonTrig-V0-wp90"
MVA_WP90 = PhoMVA_2Categories_WP(
    idName = idName,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = 0.224, # EB
    cutCategory1 = 0.355  # EE
    )

#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaPhoID_Spring15_50ns_nonTrig_V0_producer_config = cms.PSet(
    mvaName            = cms.string(mvaSpring15NonTrigClassName),
    mvaTag             = cms.string(mvaTag),
    weightFileNames    = mvaSpring15NonTrigWeightFiles_V0,
    variableDefinition  = cms.string(mvaVariablesFile),
    # Category parameters
    nCategories         = cms.int32(2),
    categoryCuts        = category_cuts
    )
# Create the VPset's for VID cuts
mvaPhoID_Spring15_50ns_nonTrig_V0_wp90 = configureVIDMVAPhoID_V1( MVA_WP90 )

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to
# 1) comment out the lines below about the registry,
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register( mvaPhoID_Spring15_50ns_nonTrig_V0_wp90.idName,
                              'f7632ecc85a3b775335fd9bf78f468df')

mvaPhoID_Spring15_50ns_nonTrig_V0_wp90.isPOGApproved = cms.untracked.bool(False)
