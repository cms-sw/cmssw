from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_tools import *

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

#
# The following MVA is derived for Fall17 samples for non-triggering photons.
# See more documentation in these presentations:
# https://indico.cern.ch/event/662751/contributions/2778043/attachments/1562017/2459674/EGamma_WorkShop_21.11.17_Debabrata.pdf
#

# This MVA implementation class name
mvaFall17v1ClassName = "PhotonMVAEstimatorRunIIFall17"
# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "v1"

# There are 2 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0    barrel photons
#   1    endcap photons

mvaRunIIFall17WeightFiles_V1 = cms.vstring(
    os.path.join(weightFileBaseDir, "Fall17/EB_V1.weights.xml.gz"),
    os.path.join(weightFileBaseDir, "Fall17/EE_V1.weights.xml.gz"),
    )

effAreasPath_pho = "RecoEgamma/PhotonIdentification/data/Fall17/effAreaPhotons_cone03_pfPhotons_90percentBased_TrueVtx.txt"
    
# The locatoins of value maps with the actual MVA values and categories
# for all particles.
# The names for the maps are "<module name>:<MVA class name>Values" 
# and "<module name>:<MVA class name>Categories"
mvaProducerModuleLabel = "photonMVAValueMapProducer"
mvaValueMapName        = mvaProducerModuleLabel + ":" + mvaFall17v1ClassName + mvaTag + "Values"
mvaCategoriesMapName   = mvaProducerModuleLabel + ":" + mvaFall17v1ClassName + mvaTag + "Categories"

# The working point for this MVA that is expected to have about 90% signal
# efficiency in each category for photons with pt>30 GeV (somewhat lower
# for lower pt photons).
idName = "mvaPhoID-RunIIFall17-v1-wp90"
MVA_WP90 = PhoMVA_2Categories_WP(
    idName = idName,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 =  0.27,  # EB new val : sig eff = 90% , bkg eff = ?
    cutCategory1 =  0.14   # EE new val : sig eff = 90% , bkg eff = ?
    )

# The working point for this MVA that is expected to have about 90% signal
# efficiency in each category for photons with pt>30 GeV (somewhat lower
# for lower pt photons).
idName = "mvaPhoID-RunIIFall17-v1-wp80"
MVA_WP80 = PhoMVA_2Categories_WP(
    idName = idName,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = 0.67,  # EB new val : sig eff = 80% , bkg eff = ?
    cutCategory1 = 0.54   # EE new val : sig eff = 80% , bkg eff = ?
    )

#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaPhoID_RunIIFall17_v1_producer_config = cms.PSet( 
    mvaName            = cms.string(mvaFall17v1ClassName),
    mvaTag             = cms.string(mvaTag),
    weightFileNames    = mvaRunIIFall17WeightFiles_V1,
    #
    # All the event content needed for this MVA implementation follows
    #
    # All the value maps: these are expected to be produced by the
    # PhotonIDValueMapProducer running upstream
    #
    phoChargedIsolation = cms.InputTag("photonIDValueMapProducer:phoChargedIsolation"),
    phoPhotonIsolation  = cms.InputTag("photonIDValueMapProducer:phoPhotonIsolation"),
    phoWorstChargedIsolation = cms.InputTag("photonIDValueMapProducer:phoWorstChargedIsolationWithPVConstraint"),
    #
    # Original event content: pileup in this case
    # 
    rho                       = cms.InputTag("fixedGridRhoAll"), # As used by Hgg and by developer of this ID
    # In this MVA for endcap the corrected photon isolation is defined as
    # iso = max( photon_isolation_raw - rho*effArea - coeff*pt, cutoff)
    # as discussed in the indico presentations listed in the beginning of this file.
    #
    effAreasConfigFile = cms.FileInPath(effAreasPath_pho),
    # The coefficients "coeff" for the formula above for linear pt scaling correction
    # the first value is for EB, the second is for EE
    # NOTE: even though the EB coefficient is provided, it is not presently used in the MVA.
    # For EB, the uncorrected raw photon isolation is used instead.
    phoIsoPtScalingCoeff = cms.vdouble(0.0035,0.0040)
    # The cutoff for the formula above
    # phoIsoCutoff = cms.double(2.5)
    )

# Create the VPset's for VID cuts
mvaPhoID_RunIIFall17_v1_wp90 = configureVIDMVAPhoID_V1( MVA_WP90 )
mvaPhoID_RunIIFall17_v1_wp80 = configureVIDMVAPhoID_V1( MVA_WP80 )

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#


central_id_registry.register( mvaPhoID_RunIIFall17_v1_wp90.idName,
                              '834dd792692b6a62786bd1caa6b53a68')
central_id_registry.register( mvaPhoID_RunIIFall17_v1_wp80.idName,
                              '7e48c47329d7d1eb889100ed03a02ba9')

mvaPhoID_RunIIFall17_v1_wp90.isPOGApproved = cms.untracked.bool(True)
mvaPhoID_RunIIFall17_v1_wp80.isPOGApproved = cms.untracked.bool(True)
