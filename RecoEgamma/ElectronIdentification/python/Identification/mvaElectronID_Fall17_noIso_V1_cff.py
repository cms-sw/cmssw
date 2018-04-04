import FWCore.ParameterSet.Config as cms

# Documentation of the MVA
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/MultivariateElectronIdentificationRun2
# https://rembserj.web.cern.ch/rembserj/notes/Electron_MVA_ID_2017_documentation

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

# This MVA implementation class name
mvaFall17ClassName = "ElectronMVAEstimatorRun2Fall17NoIso"
# The tag is an extra string attached to the names of the products
# such as ValueMaps that needs to distinguish cases when the same MVA estimator
# class is used with different tuning/weights
mvaTag = "V1"

# The parameters according to which the training bins are split:
ptSplit = 10.      # we have above and below 10 GeV categories
ebSplit = 0.800    # barrel is split into two regions
ebeeSplit = 1.479  # division between barrel and endcap

# There are 6 categories in this MVA. They have to be configured in this strict order
# (cuts and weight files order):
#   0   EB1 (eta<0.8)  pt 5-10 GeV     |   pt < ptSplit && |eta| < ebSplit
#   1   EB2 (eta>=0.8) pt 5-10 GeV     |   pt < ptSplit && |eta| >= ebSplit && |eta| < ebeeSplit
#   2   EE             pt 5-10 GeV     |   pt < ptSplit && |eta| >= ebeeSplit
#   3   EB1 (eta<0.8)  pt 10-inf GeV   |   pt >= ptSplit && |eta| < ebSplit
#   4   EB2 (eta>=0.8) pt 10-inf GeV   |   pt >= ptSplit && |eta| >= ebSplit && |eta| < ebeeSplit
#   5   EE             pt 10-inf GeV   |   pt >= ptSplit && |eta| >= ebeeSplit


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
    import (EleMVA_WP,
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
MVA_WP90 = EleMVA_WP(
    idName = idName90,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0_C0 = 0.9165112826974601, # EB1 low pt
    cutCategory0_C1 = 2.7381703555094217,
    cutCategory0_C2 = 1.03549199648109,
    cutCategory1_C0 = 0.8655738322220173, # EB2 low pt
    cutCategory1_C1 = 2.4027944652597073,
    cutCategory1_C2 = 0.7975615613282494,
    cutCategory2_C0 = -3016.035055227131, # EE low pt
    cutCategory2_C1 = -52140.61856333602,
    cutCategory2_C2 = -3016.3029387236506,
    cutCategory3_C0 = 0.9616542816132922, # EB1
    cutCategory3_C1 = 8.757943837889817,
    cutCategory3_C2 = 3.1390200321591206,
    cutCategory4_C0 = 0.9319258011430132, # EB2
    cutCategory4_C1 = 8.846057432565809,
    cutCategory4_C2 = 3.5985063793347787,
    cutCategory5_C0 = 0.8899260780999244, # EE
    cutCategory5_C1 = 10.124234115859881,
    cutCategory5_C2 = 4.352791250718547
    )

idName80 = "mvaEleID-Fall17-noIso-V1-wp80"
MVA_WP80 = EleMVA_WP(
    idName = idName80,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0_C0 = 0.9530240956555949, # EB1 low pt
    cutCategory0_C1 = 2.7591425841003647,
    cutCategory0_C2 = 0.4669644718545271,
    cutCategory1_C0 = 0.9336564763961019, # EB2 low pt
    cutCategory1_C1 = 2.709276284272272,
    cutCategory1_C2 = 0.33512286599215946,
    cutCategory2_C0 = 0.9313133688365339, # EE low pt
    cutCategory2_C1 = 1.5821934800715558,
    cutCategory2_C2 = 3.8889462619659265,
    cutCategory3_C0 = 0.9825268564943458, # EB1
    cutCategory3_C1 = 8.702601455860762,
    cutCategory3_C2 = 1.1974861596609097,
    cutCategory4_C0 = 0.9727509457929913, # EB2
    cutCategory4_C1 = 8.179525631018565,
    cutCategory4_C2 = 1.7111755094657688,
    cutCategory5_C0 = 0.9562619539540145, # EE
    cutCategory5_C1 = 8.109845366281608,
    cutCategory5_C2 = 3.013927699126942
)

### WP tuned for HZZ analysis with very high efficiency (about 98%)
# The working points were found by requiring the same signal efficiencies in
# each category as for the Spring 16 HZZ ID
# (see RecoEgamma/ElectronIdentification/python/Identification/mvaElectronID_Spring16_HZZ_V1_cff.py)
idNamewpLoose = "mvaEleID-Fall17-noIso-V1-wpLoose"
MVA_WPLoose = EleMVA_WP(
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
# Configure variable names and the values they are clipped to.
# They have to appear in the same order as in the weights xml file
#

#                Name  |  Lower clip value  | upper clip value
variablesInfo = [
                 ("ele_oldsigmaietaieta"              ,  None, None),
                 ("ele_oldsigmaiphiiphi"              ,  None, None),
                 ("ele_oldcircularity"                ,   -1.,   2.),
                 ("ele_oldr9"                         ,  None,   5.),
                 ("ele_scletawidth"                   ,  None, None),
                 ("ele_sclphiwidth"                   ,  None, None),
                 ("ele_oldhe"                         ,  None, None),
                 ("ele_kfhits"                        ,  None, None),
                 ("ele_kfchi2"                        ,  None,  10.),
                 ("ele_gsfchi2"                       ,  None, 200.),
                 ("ele_fbrem"                         ,   -1., None),
                 ("ele_gsfhits"                       ,  None, None),
                 ("ele_expected_inner_hits"           ,  None, None),
                 ("ele_conversionVertexFitProbability",  None, None),
                 ("ele_ep"                            ,  None,  20.),
                 ("ele_eelepout"                      ,  None,  20.),
                 ("ele_IoEmIop"                       ,  None, None),
                 ("ele_deltaetain"                    , -0.06, 0.06),
                 ("ele_deltaphiin"                    ,  -0.6,  0.6),
                 ("ele_deltaetaseed"                  ,  -0.2,  0.2),
                 ("rho"                               ,  None, None),
                 ("ele_psEoverEraw"                   ,  None, None), # EE only
                ]

varNames, clipLower, clipUpper = [list(l) for l in zip(*variablesInfo)]
for i, x in enumerate(clipLower):
    if x == None:
        clipLower[i] = -float('Inf')
for i, x in enumerate(clipUpper):
    if x == None:
        clipUpper[i] =  float('Inf')

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
    # Category split parameters
    ptSplit            = cms.double(ptSplit),
    ebSplit            = cms.double(ebSplit),
    ebeeSplit          = cms.double(ebeeSplit),
    # Variable clipping parameters
    varNames           = cms.vstring(*varNames),
    clipLower          = cms.vdouble(*clipLower),
    clipUpper          = cms.vdouble(*clipUpper),
    #
    weightFileNames    = mvaFall17WeightFiles_V1
    )
# Create the VPset's for VID cuts
mvaEleID_Fall17_V1_wpLoose = configureVIDMVAEleID_V1( MVA_WPLoose )
mvaEleID_Fall17_V1_wp90 = configureVIDMVAEleID_V1( MVA_WP90, cutName="GsfEleMVAExpoScalingCut")
mvaEleID_Fall17_V1_wp80 = configureVIDMVAEleID_V1( MVA_WP80, cutName="GsfEleMVAExpoScalingCut")

mvaEleID_Fall17_V1_wpLoose.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_V1_wp90.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_V1_wp80.isPOGApproved = cms.untracked.bool(True)
