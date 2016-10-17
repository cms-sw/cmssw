from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

import FWCore.ParameterSet.Config as cms

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

#
# The following MVA is tuned on Spring16 MC samples using non-triggering electrons.
# See more documentation in this presentation (P.Pigard):
#     https://indico.cern.ch/event/482674/contributions/2206032/attachments/1292177/1931287/20160621_EGM_cms_week_v5.pdf
#

# This MVA implementation class name
mvaSpring16ClassName = "ElectronMVAEstimatorRun2Spring16"
# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "V1"

# There are 6 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0   EB1 (eta<0.8)  pt 5-10 GeV
#   1   EB2 (eta>=0.8) pt 5-10 GeV
#   2   EE             pt 5-10 GeV
#   3   EB1 (eta<0.8)  pt 10-inf GeV
#   4   EB2 (eta>=0.8) pt 10-inf GeV
#   5   EE             pt 10-inf GeV

mvaSpring16WeightFiles_V1 = cms.vstring(
    "RecoEgamma/ElectronIdentification/Spring16/electronID_mva_Spring16_EB1_5_V1.weights.xml",
    "RecoEgamma/ElectronIdentification/Spring16/electronID_mva_Spring16_EB2_5_V1.weights.xml",
    "RecoEgamma/ElectronIdentification/Spring16/electronID_mva_Spring16_EE_5_V1.weights.xml",
    "RecoEgamma/ElectronIdentification/Spring16/electronID_mva_Spring16_EB1_10_V1.weights.xml",
    "RecoEgamma/ElectronIdentification/Spring16/electronID_mva_Spring16_EB2_10_V1.weights.xml",
    "RecoEgamma/ElectronIdentification/Spring16/electronID_mva_Spring16_EE_10_V1.weights.xml"
    )

# Load some common definitions for MVA machinery
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *

# The locatoins of value maps with the actual MVA values and categories
# for all particles.
# The names for the maps are "<module name>:<MVA class name>Values" 
# and "<module name>:<MVA class name>Categories"
mvaProducerModuleLabel = "electronMVAValueMapProducer"
mvaValueMapName        = mvaProducerModuleLabel + ":" + mvaSpring16ClassName + mvaTag + "Values"
mvaCategoriesMapName   = mvaProducerModuleLabel + ":" + mvaSpring16ClassName + mvaTag + "Categories"

# The working point for this MVA that is expected to have about 90% signal
# efficiency in each category
idName90 = "mvaEleID-Spring16-V1-wp90"
MVA_WP90 = EleMVA_6Categories_WP(
    idName = idName90,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = 1.1, # EB1 low pt  0.340477854013
    cutCategory1 = 1.1, # EB2 low pt 0.163703802228
    cutCategory2 = 1.1, # EE low pt 0.10562511608
    cutCategory3 = 0.802206939459, # EB1       
    cutCategory4 = 0.720191025734, # EB2       
    cutCategory5 = 0.442796376348  # EE        
    )

idName80 = "mvaEleID-Spring16-V1-wp80"
MVA_WP80 = EleMVA_6Categories_WP(
    idName = idName80,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = 1.1, # EB1 low pt 0.656394422054
    cutCategory1 = 1.1, # EB2 low pt 0.56094340086
    cutCategory2 = 1.1, # EE low pt 0.393607741594
    cutCategory3 =  0.938146269321, # EB1       
    cutCategory4 =  0.919186663628, # EB2       
    cutCategory5 =  0.822673904896  # EE        
   )

### WP tuned for HZZ analysis with very high efficiency (about 98%)
idNameLoose = "mvaEleID-Spring16-V1-wpLoose"
MVA_WPLoose = EleMVA_6Categories_WP(
    idName = idNameLoose,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 =  -0.211, # EB1 low pt
    cutCategory1 =  -0.396, # EB2 low pt
    cutCategory2 =  -0.215, # EE low pt
    cutCategory3 =  -0.870, # EB1
    cutCategory4 =  -0.838, # EB2
    cutCategory5 =  -0.763  # EE
    )


#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaEleID_Spring16_V1_producer_config = cms.PSet( 
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
mvaEleID_Spring16_V1_wpLoose = configureVIDMVAEleID_V1( MVA_WPLoose )
mvaEleID_Spring16_V1_wp90    = configureVIDMVAEleID_V1( MVA_WP90 )
mvaEleID_Spring16_V1_wp80    = configureVIDMVAEleID_V1( MVA_WP80 )


# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(mvaEleID_Spring16_V1_wpLoose.idName,
                             'c42cebb75c255a47784be47c27befc81')
central_id_registry.register(mvaEleID_Spring16_V1_wp90.idName,
                             '90a019477e7081d2bf0c9b57a389e79d')
central_id_registry.register(mvaEleID_Spring16_V1_wp80.idName,
                            'c66cebeb9db97bc1e3892008ecd1e180')

mvaEleID_Spring16_V1_wpLoose.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Spring16_V1_wp90.isPOGApproved    = cms.untracked.bool(True)
mvaEleID_Spring16_V1_wp80.isPOGApproved    = cms.untracked.bool(True)
