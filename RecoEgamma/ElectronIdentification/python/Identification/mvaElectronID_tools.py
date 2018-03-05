import FWCore.ParameterSet.Config as cms

# =========================
# Various commom parameters
# =========================

# This MVA implementation class name
mvaClassName = "ElectronMVAEstimatorRun2"

# The locatoins of value maps with the actual MVA values and categories
# for all particles.
# The names for the maps are "<module name>:<MVA class name>Values"
# and "<module name>:<MVA class name>Categories"
mvaProducerModuleLabel = "electronMVAValueMapProducer"

# The files with the variable definitions
mvaVariablesFile        = "RecoEgamma/ElectronIdentification/data/ElectronMVAEstimatorRun2Variables.txt"
mvaVariablesFileClassic = "RecoEgamma/ElectronIdentification/data/ElectronMVAEstimatorRun2VariablesClassic.txt"

# =======================================
# Define some commonly used category cuts
# =======================================

EleMVA_3CategoriesCuts = cms.vstring(
    "abs(superCluster.eta) < 0.800",
    "abs(superCluster.eta) >= 0.800 && abs(superCluster.eta) < 1.479",
    "abs(superCluster.eta) >= 1.479"
    )

EleMVA_6CategoriesCuts = cms.vstring(
    "pt < 10. && abs(superCluster.eta) < 0.800",
    "pt < 10. && abs(superCluster.eta) >= 0.800 && abs(superCluster.eta) < 1.479",
    "pt < 10. && abs(superCluster.eta) >= 1.479",
    "pt >= 10. && abs(superCluster.eta) < 0.800",
    "pt >= 10. && abs(superCluster.eta) >= 0.800 && abs(superCluster.eta) < 1.479",
    "pt >= 10. && abs(superCluster.eta) >= 1.479",
    )

# =======================================================
# Define simple containers for MVA cut values and related
# =======================================================

class EleMVA_WP:
    """
    This is a container class to hold MVA cut values for a n-category MVA
    as well as the names of the value maps that contain the MVA values computed
    for all particles in a producer upstream.

    IMPORTANT: the cuts need to be given in alphabetical order, which must
    be the order in which they are used by the cut class.
    """
    def __init__(self,
                 idName,
                 mvaTag,
                 **cuts
                 ):
        self.idName               = idName
        # map with MVA values for all particles
        self.mvaValueMapName      = mvaProducerModuleLabel + ":" + mvaClassName + mvaTag + "Values"
        # map with category index for all particles
        self.mvaCategoriesMapName = mvaProducerModuleLabel + ":" + mvaClassName + mvaTag + "Categories"
        self.cuts = cuts

    def getCutStrings(self):
        keylist = sorted(self.cuts.keys())
        return [self.cuts[key] for key in keylist]

class EleMVARaw_WP:
    """
    This is a container class to hold MVA cut values for a n-category MVA
    as well as the names of the value maps that contain the MVA values computed
    for all particles in a producer upstream.

    IMPORTANT: the cuts need to be given in alphabetical order, which must
    be the order in which they are used by the cut class.
    """
    def __init__(self,
                 idName,
                 mvaTag,
                 **cuts
                 ):
        self.idName               = idName
        # map with MVA values for all particles
        self.mvaValueMapName      = mvaProducerModuleLabel + ":" + mvaClassName + mvaTag + "RawValues"
        # map with category index for all particles
        self.mvaCategoriesMapName = mvaProducerModuleLabel + ":" + mvaClassName + mvaTag + "Categories"
        self.cuts = cuts

    def getCutStrings(self):
        keylist = self.cuts.keys()
        keylist.sort()
        return [self.cuts[key] for key in keylist]

# ================================
# Define the complete MVA cut sets
# ================================

def configureVIDMVAEleID(mvaWP, cutName="GsfEleMVACut"):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: an object of the class EleMVA_WP or similar
    that contains all necessary parameters for this MVA.
    """
    pSet = cms.PSet(
        #
        idName = cms.string( mvaWP.idName ),
        cutFlow = cms.VPSet(
            cms.PSet( cutName = cms.string(cutName),
                      mvaCuts = cms.vstring( mvaWP.getCutStrings() ),
                      mvaValueMapName = cms.InputTag( mvaWP.mvaValueMapName ),
                      mvaCategoriesMapName = cms.InputTag( mvaWP.mvaCategoriesMapName ),
                      needsAdditionalProducts = cms.bool(True),
                      isIgnored = cms.bool(False)
                      )
            )
        )
    #
    return pSet
