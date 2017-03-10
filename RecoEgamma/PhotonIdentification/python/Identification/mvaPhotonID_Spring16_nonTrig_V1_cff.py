from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

#
# The following MVA is derived for Spring16 MC samples for non-triggering photons.
# See more documentation in these presentations:
#    https://indico.cern.ch/event/491509/contributions/2226579/attachments/1303047/1946168/EGamma_PhoID_Update_IKucher.pdf
#    https://indico.cern.ch/event/578399/contributions/2344916/attachments/1357735/2053119/EGamma_PhoID_Update_19_10_2016.pdf
#

# This MVA implementation class name
mvaSpring16NonTrigClassName = "PhotonMVAEstimatorRun2Spring16NonTrig"
# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "V1"

# There are 2 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0    barrel photons
#   1    endcap photons

mvaSpring16NonTrigWeightFiles_V1 = cms.vstring(
    "RecoEgamma/PhotonIdentification/data/Spring16/photon_general_MVA_Spring16_EB_V1.weights.xml",
    "RecoEgamma/PhotonIdentification/data/Spring16/photon_general_MVA_Spring16_EE_V1.weights.xml"
    )

# Load some common definitions for MVA machinery
from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_tools import *

# The locatoins of value maps with the actual MVA values and categories
# for all particles.
# The names for the maps are "<module name>:<MVA class name>Values" 
# and "<module name>:<MVA class name>Categories"
mvaProducerModuleLabel = "photonMVAValueMapProducer"
mvaValueMapName        = mvaProducerModuleLabel + ":" + mvaSpring16NonTrigClassName + mvaTag + "Values"
mvaCategoriesMapName   = mvaProducerModuleLabel + ":" + mvaSpring16NonTrigClassName + mvaTag + "Categories"

# The working point for this MVA that is expected to have about 90% signal
# efficiency in each category for photons with pt>30 GeV (somewhat lower
# for lower pt photons).
idName = "mvaPhoID-Spring16-nonTrig-V1-wp90"
MVA_WP90 = PhoMVA_2Categories_WP(
    idName = idName,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = 0.46, # EB new val : sig eff = 90% , bkg eff = 4.6 %
    cutCategory1 = -0.12  # EE new val : sig eff = 90% , bkg eff = 6%
    )

#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaPhoID_Spring16_nonTrig_V1_producer_config = cms.PSet( 
    mvaName            = cms.string(mvaSpring16NonTrigClassName),
    mvaTag             = cms.string(mvaTag),
    weightFileNames    = mvaSpring16NonTrigWeightFiles_V1,
    #
    # All the event content needed for this MVA implementation follows
    #
    # All the value maps: these are expected to be produced by the
    # PhotonIDValueMapProducer running upstream
    #
    phoChargedIsolation = cms.InputTag("egmPhotonIsolation:h+-DR030-"),
    phoPhotonIsolation  = cms.InputTag("egmPhotonIsolation:gamma-DR030-"),
    phoWorstChargedIsolation = cms.InputTag("photonIDValueMapProducer:phoWorstChargedIsolation"),
    #
    # Original event content: pileup in this case
    # 
    rho                       = cms.InputTag("fixedGridRhoFastjetAll") 
    )
# Create the VPset's for VID cuts
mvaPhoID_Spring16_nonTrig_V1_wp90 = configureVIDMVAPhoID_V1( MVA_WP90 )

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register( mvaPhoID_Spring16_nonTrig_V1_wp90.idName,
                              '0db3574f257ba9082ffc4874ba6cee99')

mvaPhoID_Spring16_nonTrig_V1_wp90.isPOGApproved = cms.untracked.bool(True)
