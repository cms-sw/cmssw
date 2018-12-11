from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *
import FWCore.ParameterSet.Config as cms

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

#
# The following MVA is tuned on Spring16 MC samples using non-triggering electrons.
# This is ID only provides a very loose (~98% signal efficiency) working point and 
# is intended to be used by multi-lepton analyses (HZZ, ZZ, ... ). Please consider 
# using the general purpose ID.
# See more documentation in this presentation (P.Pigard):
#     https://indico.cern.ch/event/482674/contributions/2206032/attachments/1292177/1931287/20160621_EGM_cms_week_v5.pdf
#

# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "Spring16HZZV1"

# There are 6 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0   EB1 (eta<0.8)  pt 5-10 GeV
#   1   EB2 (eta>=0.8) pt 5-10 GeV
#   2   EE             pt 5-10 GeV
#   3   EB1 (eta<0.8)  pt 10-inf GeV
#   4   EB2 (eta>=0.8) pt 10-inf GeV
#   5   EE             pt 10-inf GeV

mvaSpring16WeightFiles_V1 = cms.vstring(
    "RecoEgamma/ElectronIdentification/data/Spring16_HZZ_V1/electronID_mva_Spring16_HZZ_V1_EB1_5.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring16_HZZ_V1/electronID_mva_Spring16_HZZ_V1_EB2_5.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring16_HZZ_V1/electronID_mva_Spring16_HZZ_V1_EE_5.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring16_HZZ_V1/electronID_mva_Spring16_HZZ_V1_EB1_10.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring16_HZZ_V1/electronID_mva_Spring16_HZZ_V1_EB2_10.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Spring16_HZZ_V1/electronID_mva_Spring16_HZZ_V1_EE_10.weights.xml.gz"
    )

### WP tuned for HZZ analysis with very high efficiency (about 98%)
idNameLoose = "mvaEleID-Spring16-HZZ-V1-wpLoose"
MVA_WPLoose = EleMVA_WP(
    idNameLoose, mvaTag,
    cutCategory0 =  "-0.211", # EB1 low pt
    cutCategory1 =  "-0.396", # EB2 low pt
    cutCategory2 =  "-0.215", # EE low pt
    cutCategory3 =  "-0.870", # EB1
    cutCategory4 =  "-0.838", # EB2
    cutCategory5 =  "-0.763"  # EE
    )

workingPoints = dict(
    wpLoose = MVA_WPLoose
)


#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaEleID_Spring16_HZZ_V1_producer_config = cms.PSet(
    mvaName            = cms.string(mvaClassName),
    mvaTag             = cms.string(mvaTag),
    # Category parameters
    nCategories         = cms.int32(6),
    categoryCuts        = cms.vstring(*EleMVA_6CategoriesCuts),
    # Weight files and variable definitions
    weightFileNames     = mvaSpring16WeightFiles_V1,
    variableDefinition  = cms.string(mvaVariablesFile)
    )
# Create the VPset's for VID cuts
mvaEleID_Spring16_HZZ_V1_wpLoose = configureVIDMVAEleID( MVA_WPLoose )


# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to 
# 1) comment out the lines below about the registry, 
# 2) run "calculateMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register(mvaEleID_Spring16_HZZ_V1_wpLoose.idName,
                             '1797cc03eb62387e10266fca72ea10cd')

mvaEleID_Spring16_HZZ_V1_wpLoose.isPOGApproved = cms.untracked.bool(True)
