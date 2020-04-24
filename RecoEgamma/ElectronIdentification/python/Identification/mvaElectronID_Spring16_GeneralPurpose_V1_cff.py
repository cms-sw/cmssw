from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

#
# The following MVA is tuned on Spring16 MC samples.
# See more documentation in this presentation (P.Pigard):
#    https://indico.cern.ch/event/491544/contributions/2321565/attachments/1346333/2030225/20160929_EGM_v4.pdf 
#

# This MVA implementation class name
mvaSpring16ClassName = "ElectronMVAEstimatorRun2Spring16GeneralPurpose"
# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "V1"

# There are 3 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0   EB1 (eta<0.8)  pt 10-inf GeV
#   1   EB2 (eta>=0.8) pt 10-inf GeV
#   2   EE             pt 10-inf GeV

mvaSpring16WeightFiles_V1 = cms.vstring(
    "RecoEgamma/ElectronIdentification/data/Spring16_GeneralPurpose_V1/electronID_mva_Spring16_GeneralPurpose_V1_EB1_10.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring16_GeneralPurpose_V1/electronID_mva_Spring16_GeneralPurpose_V1_EB2_10.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring16_GeneralPurpose_V1/electronID_mva_Spring16_GeneralPurpose_V1_EE_10.weights.xml.gz"
    )

# Load some common definitions for MVA machinery
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools \
    import (EleMVA_3Categories_WP,
            configureVIDMVAEleID_V1 )

# The locatoins of value maps with the actual MVA values and categories
# for all particles.
# The names for the maps are "<module name>:<MVA class name>Values" 
# and "<module name>:<MVA class name>Categories"
mvaProducerModuleLabel = "electronMVAValueMapProducer"
mvaValueMapName        = mvaProducerModuleLabel + ":" + mvaSpring16ClassName + mvaTag + "Values"
mvaCategoriesMapName   = mvaProducerModuleLabel + ":" + mvaSpring16ClassName + mvaTag + "Categories"

### WP to give about 90 and 80% signal efficiecny for electrons from Drell-Yan with pT > 25 GeV
### For turn-on and details see documentation linked above
MVA_WP90 = EleMVA_3Categories_WP(
    idName = "mvaEleID-Spring16-GeneralPurpose-V1-wp90",
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 =  0.836695742607, # EB1
    cutCategory1 =  0.715337944031, # EB2
    cutCategory2 =  0.356799721718, # EE
    )
MVA_WP80 = EleMVA_3Categories_WP(
    idName = "mvaEleID-Spring16-GeneralPurpose-V1-wp80",
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 =  0.940962684155, # EB1
    cutCategory1 =  0.899208843708, # EB2
    cutCategory2 =  0.758484721184, # EE
    )



#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaEleID_Spring16_GeneralPurpose_V1_producer_config = cms.PSet( 
    mvaName            = cms.string(mvaSpring16ClassName),
    mvaTag             = cms.string(mvaTag),
    # This MVA uses conversion info, so configure several data items on that
    beamSpot           = cms.InputTag('offlineBeamSpot'),
    conversionsAOD     = cms.InputTag('allConversions'),
    conversionsMiniAOD = cms.InputTag('reducedEgamma:reducedConversions'),
    #
    weightFileNames    = mvaSpring16WeightFiles_V1
    )
# Create the VPset's for VID cuts
mvaEleID_Spring16_GeneralPurpose_V1_wp90 = configureVIDMVAEleID_V1( MVA_WP90 )
mvaEleID_Spring16_GeneralPurpose_V1_wp80 = configureVIDMVAEleID_V1( MVA_WP80 )


# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(mvaEleID_Spring16_GeneralPurpose_V1_wp90.idName,
                             '14c153aaf3c207deb3ad4932586647a7')
central_id_registry.register(mvaEleID_Spring16_GeneralPurpose_V1_wp80.idName,
                             'b490bc0b0af2d5f3e9efea562370af2a')


mvaEleID_Spring16_GeneralPurpose_V1_wp90.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Spring16_GeneralPurpose_V1_wp80.isPOGApproved = cms.untracked.bool(True)
