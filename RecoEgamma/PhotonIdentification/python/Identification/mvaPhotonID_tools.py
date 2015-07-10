import FWCore.ParameterSet.Config as cms

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
