import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry
from os import path

weightFileBaseDir = "RecoEgamma/PhotonIdentification/data/MVA"

# division between barrel and endcap
ebeeSplit = 1.479
# categories
category_cuts = cms.vstring(
    "abs(superCluster.eta) <  1.479",
    "abs(superCluster.eta) >= 1.479",
    )

# This MVA implementation class name
mvaClassName = "PhotonMVAEstimator"

# The locatoins of value maps with the actual MVA values and categories
# for all particles.
# The names for the maps are "<module name>:<MVA class name>Values" 
# and "<module name>:<MVA class name>Categories"
mvaProducerModuleLabel = "photonMVAValueMapProducer"

# =======================================================
# Define simple containers for MVA cut values and related
# =======================================================

class PhoMVA_2Categories_WP:
    """
    This is a container class to hold MVA cut values for a 2-category MVA
    as well as the names of the value maps that contain the MVA values computed
    for all particles in a producer upstream.
    """
    def __init__(self,
                 idName,
                 mvaValueMapName,
                 mvaCategoriesMapName,
                 cutCategory0,
                 cutCategory1
                 ):
        self.idName       = idName
        self.mvaValueMapName      = mvaValueMapName
        self.mvaCategoriesMapName = mvaCategoriesMapName
        self.cutCategory0 = cutCategory0
        self.cutCategory1 = cutCategory1

    def getCutValues(self):
        return [self.cutCategory0, self.cutCategory1]

# ==============================================================
# Define the complete MVA cut sets
# ==============================================================
    
def configureVIDMVAPhoID_V1( mvaWP ):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: an object of the class PhoMVA_2Categories_WP or similar
    that contains all necessary parameters for this MVA.
    """
    parameterSet =  cms.PSet(
        #
        idName = cms.string( mvaWP.idName ), 
        cutFlow = cms.VPSet( 
            cms.PSet( cutName = cms.string("PhoMVACut"),
                      mvaCuts = cms.vdouble( mvaWP.getCutValues() ),
                      mvaValueMapName = cms.InputTag( mvaWP.mvaValueMapName ),
                      mvaCategoriesMapName =cms.InputTag( mvaWP.mvaCategoriesMapName ),
                      needsAdditionalProducts = cms.bool(True),
                      isIgnored = cms.bool(False)
                      )
            )
        )
    #
    return parameterSet

# ===============================================
# High level function to create a two category ID
# ===============================================

# mvaTag:
#         The mvaTag is an extra string attached to the names of the products
#         such as ValueMaps that needs to distinguish cases when the same MVA estimator
#         class is used with different tuning/weights
# variablesFile:
#         The file listing the variables used in this MVA
# weightFiles:
#         The weight files in the order EB first, then EE
# wpConfig:
#         A dictionary with the names and cut values of the working points
# addKwargsForValueProducer:
#         Additional keyword parameters passed to the producer config

def configureFullVIDMVAPhoID(mvaTag, variablesFile, weightFiles, wpConfig, **addKwargsForValueProducer):

    mvaValueMapName        = mvaProducerModuleLabel + ":" + mvaClassName + mvaTag + "Values"
    mvaCategoriesMapName   = mvaProducerModuleLabel + ":" + mvaClassName + mvaTag + "Categories"

    # Create the PSet that will be fed to the MVA value map producer
    producer_config = cms.PSet( 
        mvaName             = cms.string(mvaClassName),
        mvaTag              = cms.string(mvaTag),
        weightFileNames     = cms.vstring(*weightFiles),
        variableDefinition  = cms.string(variablesFile),
        **addKwargsForValueProducer
        )

    # Create the VPset's for VID cuts
    VID_config = {}
    for wpc in wpConfig:
        idName = wpc["idName"]
        VID_config[idName] = configureVIDMVAPhoID_V1(
                PhoMVA_2Categories_WP(
                    idName = idName,
                    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
                    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
                    cutCategory0 =  wpc["cuts"]["EB"],  # EB
                    cutCategory1 =  wpc["cuts"]["EE"]   # EE
                )
            )

    configs = {"producer_config": producer_config, "VID_config": VID_config}
    return configs
