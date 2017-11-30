import FWCore.ParameterSet.Config as cms

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

# This MVA implementation class name
mvaFall17ClassName = "ElectronMVAEstimatorRun2Fall17noIso"
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

mvaFall17WeightFiles_V1 = cms.vstring(
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB1_5_2017_puinfo_BDT.weights.xml",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB2_5_2017_puinfo_BDT.weights.xml",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EE_5_2017_puinfo_BDT.weights.xml",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB1_10_2017_puinfo_BDT.weights.xml",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB2_10_2017_puinfo_BDT.weights.xml",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EE_10_2017_puinfo_BDT.weights.xml"
    )

# Load some common definitions for MVA machinery
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools \
    import (EleMVA_6Categories_WP,
            configureVIDMVAEleID_V1)

# The locatoins of value maps with the actual MVA values and categories
# for all particles.
# The names for the maps are "<module name>:<MVA class name>Values"
# and "<module name>:<MVA class name>Categories"
mvaProducerModuleLabel = "electronMVAValueMapProducer"
mvaValueMapName        = mvaProducerModuleLabel + ":" + mvaFall17ClassName + mvaTag + "Values"
mvaCategoriesMapName   = mvaProducerModuleLabel + ":" + mvaFall17ClassName + mvaTag + "Categories"

## The working point for this MVA that is expected to have about 90% signal
# WP tuned to give about 90 and 80% signal efficiecny for electrons from Drell-Yan with pT > 25 GeV
# The working point for the low pt categories is just taken over from the high pt
idName90 = "mvaEleID-Fall17-noIso-V1-wp90"
MVA_WP90 = EleMVA_6Categories_WP(
    idName = idName90,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = 0.9096365928649902, # EB1 low pt
    cutCategory1 = 0.8497519493103027, # EB2 low pt
    cutCategory2 = 0.7578120231628419, # EE low pt 
    cutCategory3 = 0.9096365928649902, # EB1       
    cutCategory4 = 0.8497519493103027, # EB2       
    cutCategory5 = 0.7578120231628419  # EE        
    )

idName80 = "mvaEleID-Fall17-noIso-V1-wp80"
MVA_WP80 = EleMVA_6Categories_WP(
    idName = idName80,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = 0.9630242347717283, # EB1 low pt
    cutCategory1 = 0.9461585044860840, # EB2 low pt
    cutCategory2 = 0.9160532951354982, # EE low pt 
    cutCategory3 = 0.9630242347717283, # EB1       
    cutCategory4 = 0.946158504486084 , # EB2       
    cutCategory5 = 0.9160532951354982  # EE        
)

### WP tuned for HZZ analysis with very high efficiency (about 98%)
# The working points were found by requiring the same signal efficiencies in
# each category as for the Spring 16 HZZ ID
# (see RecoEgamma/ElectronIdentification/python/Identification/mvaElectronID_Spring16_HZZ_V1_cff.py)
idNamewpLoose = "mvaEleID-Fall17-noIso-V1-wpLoose"
MVA_WPLoose = EleMVA_6Categories_WP(
    idName = idNamewpLoose,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 =  -0.13285867293779202, # EB1 low pt
    cutCategory1 =  -0.31765300958836074, # EB2 low pt
    cutCategory2 =  -0.0799205914718861 , # EE low pt
    cutCategory3 =  -0.856871961305474  , # EB1
    cutCategory4 =  -0.8107642141584835 , # EB2
    cutCategory5 =  -0.7179265933023059   # EE
    )


#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaEleID_Fall17_noIso_V1_producer_config = cms.PSet(
    mvaName            = cms.string(mvaFall17ClassName),
    mvaTag             = cms.string(mvaTag),
    # This MVA uses conversion info, so configure several data items on that
    beamSpot           = cms.InputTag('offlineBeamSpot'),
    conversionsAOD     = cms.InputTag('allConversions'),
    conversionsMiniAOD = cms.InputTag('reducedEgamma:reducedConversions'),
    #
    weightFileNames    = mvaFall17WeightFiles_V1
    )
# Create the VPset's for VID cuts
mvaEleID_Fall17_V1_wpLoose = configureVIDMVAEleID_V1( MVA_WPLoose )
mvaEleID_Fall17_V1_wp90 = configureVIDMVAEleID_V1( MVA_WP90 )
mvaEleID_Fall17_V1_wp80 = configureVIDMVAEleID_V1( MVA_WP80 )

mvaEleID_Fall17_V1_wpLoose.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_V1_wp90.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_V1_wp80.isPOGApproved = cms.untracked.bool(True)
