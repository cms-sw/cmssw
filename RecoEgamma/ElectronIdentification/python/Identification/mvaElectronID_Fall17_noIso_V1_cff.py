import FWCore.ParameterSet.Config as cms
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *

# Documentation of the MVA
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/MultivariateElectronIdentificationRun2
# https://rembserj.web.cern.ch/rembserj/notes/Electron_MVA_ID_2017_documentation

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "Fall17NoIsoV1"

# There are 6 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0   EB1 (eta<0.8)  pt 5-10 GeV     |   pt < ptSplit && |eta| < ebSplit
#   1   EB2 (eta>=0.8) pt 5-10 GeV     |   pt < ptSplit && |eta| >= ebSplit && |eta| < ebeeSplit
#   2   EE             pt 5-10 GeV     |   pt < ptSplit && |eta| >= ebeeSplit
#   3   EB1 (eta<0.8)  pt 10-inf GeV   |   pt >= ptSplit && |eta| < ebSplit
#   4   EB2 (eta>=0.8) pt 10-inf GeV   |   pt >= ptSplit && |eta| >= ebSplit && |eta| < ebeeSplit
#   5   EE             pt 10-inf GeV   |   pt >= ptSplit && |eta| >= ebeeSplit


mvaFall17WeightFiles_V1 = cms.vstring(
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB1_5_2017_puinfo_BDT.weights.root",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB2_5_2017_puinfo_BDT.weights.root",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EE_5_2017_puinfo_BDT.weights.root",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB1_10_2017_puinfo_BDT.weights.root",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB2_10_2017_puinfo_BDT.weights.root",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EE_10_2017_puinfo_BDT.weights.root"
    )

## The working point for this MVA that is expected to have about 90% signal
# WP tuned to give about 90 and 80% signal efficiecny for electrons from Drell-Yan with pT > 25 GeV
# The working point for the low pt categories is just taken over from the high pt
idName90 = "mvaEleID-Fall17-noIso-V1-wp90"
MVA_WP90 = EleMVA_WP(
    idName = idName90, mvaTag = mvaTag,
    cutCategory0 = "0.9165112826974601 - exp(-pt / 2.7381703555094217) *    1.03549199648109", # EB1 low pt
    cutCategory1 = "0.8655738322220173 - exp(-pt / 2.4027944652597073) *  0.7975615613282494", # EB2 low pt
    cutCategory2 = "-3016.035055227131 - exp(-pt / -52140.61856333602) * -3016.3029387236506", # EE low pt
    cutCategory3 = "0.9616542816132922 - exp(-pt /  8.757943837889817) *  3.1390200321591206", # EB1
    cutCategory4 = "0.9319258011430132 - exp(-pt /  8.846057432565809) *  3.5985063793347787", # EB2
    cutCategory5 = "0.8899260780999244 - exp(-pt / 10.124234115859881) *   4.352791250718547", # EE
    )

idName80 = "mvaEleID-Fall17-noIso-V1-wp80"
MVA_WP80 = EleMVA_WP(
    idName = idName80, mvaTag = mvaTag,
    cutCategory0 = "0.9530240956555949 - exp(-pt / 2.7591425841003647) *  0.4669644718545271", # EB1 low pt
    cutCategory1 = "0.9336564763961019 - exp(-pt /  2.709276284272272) * 0.33512286599215946", # EB2 low pt
    cutCategory2 = "0.9313133688365339 - exp(-pt / 1.5821934800715558) *  3.8889462619659265", # EE low pt
    cutCategory3 = "0.9825268564943458 - exp(-pt /  8.702601455860762) *  1.1974861596609097", # EB1
    cutCategory4 = "0.9727509457929913 - exp(-pt /  8.179525631018565) *  1.7111755094657688", # EB2
    cutCategory5 = "0.9562619539540145 - exp(-pt /  8.109845366281608) *   3.013927699126942", # EE
)

### WP tuned for HZZ analysis with very high efficiency (about 98%)
# The working points were found by requiring the same signal efficiencies in
# each category as for the Spring 16 HZZ ID
# (see RecoEgamma/ElectronIdentification/python/Identification/mvaElectronID_Spring16_HZZ_V1_cff.py)
idNamewpLoose = "mvaEleID-Fall17-noIso-V1-wpLoose"
MVA_WPLoose = EleMVA_WP(
    idName = idNamewpLoose, mvaTag = mvaTag,
    cutCategory0 =  "-0.13285867293779202", # EB1 low pt
    cutCategory1 =  "-0.31765300958836074", # EB2 low pt
    cutCategory2 =  "-0.0799205914718861" , # EE low pt
    cutCategory3 =  "-0.856871961305474"  , # EB1
    cutCategory4 =  "-0.8107642141584835" , # EB2
    cutCategory5 =  "-0.7179265933023059"   # EE
    )

#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaEleID_Fall17_noIso_V1_producer_config = cms.PSet(
    mvaName             = cms.string(mvaClassName),
    mvaTag              = cms.string(mvaTag),
    # Category parameters
    nCategories         = cms.int32(6),
    categoryCuts        = cms.vstring(*EleMVA_6CategoriesCuts),
    # Weight files and variable definitions
    weightFileNames     = mvaFall17WeightFiles_V1,
    variableDefinition  = cms.string("RecoEgamma/ElectronIdentification/data/ElectronMVAEstimatorRun2Fall17V1Variables.txt")
    )
# Create the VPset's for VID cuts
mvaEleID_Fall17_V1_wpLoose = configureVIDMVAEleID( MVA_WPLoose )
mvaEleID_Fall17_V1_wp90 = configureVIDMVAEleID( MVA_WP90 )
mvaEleID_Fall17_V1_wp80 = configureVIDMVAEleID( MVA_WP80 )

mvaEleID_Fall17_V1_wpLoose.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_V1_wp90.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_V1_wp80.isPOGApproved = cms.untracked.bool(True)
