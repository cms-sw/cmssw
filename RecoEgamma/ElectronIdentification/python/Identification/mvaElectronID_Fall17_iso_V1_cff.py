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
mvaTag = "Fall17IsoV1"

# There are 6 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0   EB1 (eta<0.8)  pt 5-10 GeV     |   pt < ptSplit && |eta| < ebSplit
#   1   EB2 (eta>=0.8) pt 5-10 GeV     |   pt < ptSplit && |eta| >= ebSplit && |eta| < ebeeSplit
#   2   EE             pt 5-10 GeV     |   pt < ptSplit && |eta| >= ebeeSplit
#   3   EB1 (eta<0.8)  pt 10-inf GeV   |   pt >= ptSplit && |eta| < ebSplit
#   4   EB2 (eta>=0.8) pt 10-inf GeV   |   pt >= ptSplit && |eta| >= ebSplit && |eta| < ebeeSplit
#   5   EE             pt 10-inf GeV   |   pt >= ptSplit && |eta| >= ebeeSplit


mvaFall17WeightFiles_V1 = cms.vstring(
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB1_5_2017_puinfo_iso_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB2_5_2017_puinfo_iso_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EE_5_2017_puinfo_iso_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB1_10_2017_puinfo_iso_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB2_10_2017_puinfo_iso_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EE_10_2017_puinfo_iso_BDT.weights.xml.gz"
    )

## The working point for this MVA that is expected to have about 90% signal
# WP tuned to give about 90 and 80% signal efficiecny for electrons from Drell-Yan with pT > 25 GeV
# The working point for the low pt categories is just taken over from the high pt
idName90 = "mvaEleID-Fall17-iso-V1-wp90"
MVA_WP90 = EleMVA_WP(
    idName90, mvaTag,
    cutCategory0 = "0.9387070396095831 - exp(-pt /   2.6525585228167636) *  0.8222647164151365", # EB1 low pt
    cutCategory1 = "0.8948802925677235 - exp(-pt /   2.7645670358783523) *  0.4123381218697539", # EB2 low pt
    cutCategory2 = "-1830.8583661119892 - exp(-pt /   -36578.11055382301) * -1831.2083578116517", # EE low pt
    cutCategory3 = "0.9717674837607253 - exp(-pt /    8.912850985100356) *  1.9712414940437244", # EB1
    cutCategory4 = "0.9458745023265976 - exp(-pt /     8.83104420392795) *    2.40849932040698", # EB2
    cutCategory5 = "0.8979112012086751 - exp(-pt /    9.814082144168015) *   4.171581694893849", # EE
)

idName80 = "mvaEleID-Fall17-iso-V1-wp80"
MVA_WP80 = EleMVA_WP(
    idName80, mvaTag,
    cutCategory0 = "0.9725509559754997 - exp(-pt /  2.976593261509491) *  0.2653858736397496", # EB1 low pt
    cutCategory1 = "0.9508038141601247 - exp(-pt / 2.6633500558725713) *  0.2355820499260076", # EB2 low pt
    cutCategory2 = "0.9365037167596238 - exp(-pt / 1.5765442323949856) *   3.067015289215309", # EE low pt
    cutCategory3 = "0.9896562087723659 - exp(-pt / 10.342490511998674) * 0.40204156417414094", # EB1
    cutCategory4 = "0.9819232656533827 - exp(-pt /   9.05548836482051) *   0.772674931169389", # EB2
    cutCategory5 = "0.9625098201744635 - exp(-pt /   8.42589315557279) *  2.2916152615134173", # EE
)

### WP tuned for HZZ analysis with very high efficiency (about 98%)
# The working points were found by requiring the same signal efficiencies in
# each category as for the Spring 16 HZZ ID
# (see RecoEgamma/ElectronIdentification/python/Identification/mvaElectronID_Spring16_HZZ_V1_cff.py)
idNamewpLoose = "mvaEleID-Fall17-iso-V1-wpLoose"
MVA_WPLoose = EleMVA_WP(
    idNamewpLoose, mvaTag,
    cutCategory0 =  "-0.09564086146419018", # EB1 low pt
    cutCategory1 =  "-0.28229916981926795", # EB2 low pt
    cutCategory2 =  "-0.05466682296962322", # EE low pt
    cutCategory3 =  "-0.833466688584422"  , # EB1
    cutCategory4 =  "-0.7677000247570116" , # EB2
    cutCategory5 =  "-0.6917305995653829"   # EE
    )

#
# Finally, set up VID configuration for all cuts
#

# Create the PSet that will be fed to the MVA value map producer
mvaEleID_Fall17_iso_V1_producer_config = cms.PSet(
    mvaName             = cms.string(mvaClassName),
    mvaTag              = cms.string(mvaTag),
    # Category parameters
    nCategories         = cms.int32(6),
    categoryCuts        = EleMVA_6CategoriesCuts,
    # Weight files and variable definitions
    weightFileNames     = mvaFall17WeightFiles_V1,
    variableDefinition  = cms.string("RecoEgamma/ElectronIdentification/data/ElectronMVAEstimatorRun2Fall17V1Variables.txt")
    )
# Create the VPset's for VID cuts
mvaEleID_Fall17_V1_wpLoose = configureVIDMVAEleID( MVA_WPLoose )
mvaEleID_Fall17_V1_wp90 = configureVIDMVAEleID( MVA_WP90)
mvaEleID_Fall17_V1_wp80 = configureVIDMVAEleID( MVA_WP80)

mvaEleID_Fall17_V1_wpLoose.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_V1_wp90.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_V1_wp80.isPOGApproved = cms.untracked.bool(True)
