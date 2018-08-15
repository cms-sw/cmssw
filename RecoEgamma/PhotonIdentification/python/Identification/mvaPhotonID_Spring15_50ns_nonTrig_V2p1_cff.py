from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_tools import *

mvaVariablesFile        = "RecoEgamma/PhotonIdentification/data/PhotonMVAEstimatorRun2VariablesSpring15.txt"

# This MVA ID is the same as 50ns V2 except it uses quantities embedded
# in the objects rather than value maps.
#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

#
# The following MVA is derived for Spring15 MC samples for non-triggering photons.
# See more documentation in this presentation:
#    https://indico.cern.ch/event/369237/contribution/2/attachments/1128009/1611753/egamma-July17-2015.pdf
#    this also contains a minor update from email exchanges (thus move to V2)
#    Specific docs for V2 are in this final presentation:
#    https://indico.cern.ch/event/369241/contribution/1/attachments/1140148/1632879/egamma-Aug14-2015.pdf

# This MVA implementation class name
mvaSpring15NonTrigClassName = "PhotonMVAEstimator"
# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "Run2Spring15NonTrig50nsV2p1"

# There are 2 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0    barrel photons
#   1    endcap photons

mvaSpring15NonTrigWeightFiles_V2p1 = cms.vstring(
    path.join(weightFileBaseDir, "Spring15/50ns_EB_V2.weights.xml.gz"),
    path.join(weightFileBaseDir, "Spring15/50ns_EB_V2.weights.xml.gz"),
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
idName = "mvaPhoID-Spring15-50ns-nonTrig-V2p1-wp90"
MVA_WP90 = PhoMVA_2Categories_WP(
    idName = idName,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = 0.29538, # EB
    cutCategory1 = 0.45837  # EE
    )

#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaPhoID_Spring15_50ns_nonTrig_V2p1_producer_config = cms.PSet(
    mvaName            = cms.string(mvaSpring15NonTrigClassName),
    mvaTag             = cms.string(mvaTag),
    weightFileNames    = mvaSpring15NonTrigWeightFiles_V2p1,
    variableDefinition  = cms.string(mvaVariablesFile),
    # Category parameters
    nCategories         = cms.int32(2),
    categoryCuts        = category_cuts
    )
# Create the VPset's for VID cuts
mvaPhoID_Spring15_50ns_nonTrig_V2p1_wp90 = configureVIDMVAPhoID_V1( MVA_WP90 )

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to
# 1) comment out the lines below about the registry,
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register( mvaPhoID_Spring15_50ns_nonTrig_V2p1_wp90.idName,
                              '9b03aa4412c43fdbd72ea55f2e1a16bc')

mvaPhoID_Spring15_50ns_nonTrig_V2p1_wp90.isPOGApproved = cms.untracked.bool(True)
