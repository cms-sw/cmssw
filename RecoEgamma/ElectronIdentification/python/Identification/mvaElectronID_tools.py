import FWCore.ParameterSet.Config as cms

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
                 mvaValueMapName,
                 mvaCategoriesMapName,
                 **cuts
                 ):
        self.idName       = idName
        self.mvaValueMapName      = mvaValueMapName
        self.mvaCategoriesMapName = mvaCategoriesMapName
        self.cuts = cuts

    def getCutValues(self):
        keylist = sorted(self.cuts.keys())
        return [self.cuts[key] for key in keylist]

# This is for backwards compatibility with IDs <= 2016
EleMVA_3Categories_WP = EleMVA_WP
EleMVA_6Categories_WP = EleMVA_WP

# ==============================================================
# Define the complete MVA cut sets
# ==============================================================
    
def configureVIDMVAEleID_V1(mvaWP, cutName="GsfEleMVACut"):
    """
    This function configures the full cms.PSet for a VID ID and returns it.
    The inputs: an object of the class EleMVA_WP or similar
    that contains all necessary parameters for this MVA.
    """
    parameterSet =  cms.PSet(
        #
        idName = cms.string( mvaWP.idName ), 
        cutFlow = cms.VPSet( 
            cms.PSet( cutName = cms.string(cutName),
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
