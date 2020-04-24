import FWCore.ParameterSet.Config as cms
import os.path

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
mvaTag = "V2"

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

weightFileDir = "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/Fall17NoIsoV2"

mvaWeightFiles = cms.vstring(
     os.path.join(weightFileDir, "EB1_5.weights.xml.gz"), # EB1_5
     os.path.join(weightFileDir, "EB2_5.weights.xml.gz"), # EB2_5
     os.path.join(weightFileDir, "EE_5.weights.xml.gz"), # EE_5
     os.path.join(weightFileDir, "EB1_10.weights.xml.gz"), # EB1_10
     os.path.join(weightFileDir, "EB2_10.weights.xml.gz"), # EB2_10
     os.path.join(weightFileDir, "EE_10.weights.xml.gz"), # EE_10
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
mvaValueMapName        = mvaProducerModuleLabel + ":" + mvaFall17ClassName + mvaTag + "RawValues"
mvaCategoriesMapName   = mvaProducerModuleLabel + ":" + mvaFall17ClassName + mvaTag + "Categories"

## The working point for this MVA that is expected to have about 90% signal
# WP tuned to give about 90 and 80% signal efficiecny for electrons from Drell-Yan with pT > 25 GeV
# The working point for the low pt categories is just taken over from the high pt
idName90 = "mvaEleID-Fall17-noIso-V2-wp90"
MVA_WP90 = EleMVA_WP(
    idName = idName90,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0_C0 = 2.77072387339, # EB1 low pt
    cutCategory0_C1 = 3.81500912145,
    cutCategory0_C2 = 8.16304860178,
    cutCategory1_C0 = 1.85602317813, # EB2 low pt
    cutCategory1_C1 = 2.18697654938,
    cutCategory1_C2 = 11.8568936824,
    cutCategory2_C0 = 1.73489307814, # EE low pt
    cutCategory2_C1 = 2.0163211971,
    cutCategory2_C2 = 17.013880078,
    cutCategory3_C0 = 5.9175992258, # EB1
    cutCategory3_C1 = 13.4807294538,
    cutCategory3_C2 = 9.31966232685,
    cutCategory4_C0 = 5.01598837255, # EB2
    cutCategory4_C1 = 13.1280451502,
    cutCategory4_C2 = 8.79418193765,
    cutCategory5_C0 = 4.16921343208, # EE
    cutCategory5_C1 = 13.2017224621,
    cutCategory5_C2 = 9.00720913211
    )

idName80 = "mvaEleID-Fall17-noIso-V2-wp80"
MVA_WP80 = EleMVA_WP(
    idName = idName80,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0_C0 = 3.26449620468, # EB1 low pt
    cutCategory0_C1 = 3.32657149223,
    cutCategory0_C2 = 8.84669783568,
    cutCategory1_C0 = 2.83557838497, # EB2 low pt
    cutCategory1_C1 = 2.15150487651,
    cutCategory1_C2 = 11.0978016567,
    cutCategory2_C0 = 2.91994945177, # EE low pt
    cutCategory2_C1 = 1.69875477522,
    cutCategory2_C2 = 24.024807824,
    cutCategory3_C0 = 7.1336238874, # EB1
    cutCategory3_C1 = 16.5605268797,
    cutCategory3_C2 = 8.22531222391,
    cutCategory4_C0 = 6.18638275782, # EB2
    cutCategory4_C1 = 15.2694634284,
    cutCategory4_C2 = 7.49764565324,
    cutCategory5_C0 = 5.43175865738, # EE
    cutCategory5_C1 = 15.4290075949,
    cutCategory5_C2 = 7.56899692285
)

### WP tuned for HZZ analysis with very high efficiency (about 98%)
# The working points were found by requiring the same signal efficiencies in
# each category as for the Spring 16 HZZ ID
# (see RecoEgamma/ElectronIdentification/python/Identification/mvaElectronID_Spring16_HZZ_V2_cff.py)
idNamewpLoose = "mvaEleID-Fall17-noIso-V2-wpLoose"
MVA_WPLoose = EleMVA_WP(
    idName = idNamewpLoose,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 = 0.894411158628, # EB1_5
    cutCategory1 = 0.791966464633, # EB2_5
    cutCategory2 = 1.47104857173, # EE_5
    cutCategory3 = -0.293962958665, # EB1_10
    cutCategory4 = -0.250424758584, # EB2_10
    cutCategory5 = -0.130985179031, # EE_10
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
mvaEleID_Fall17_noIso_V2_producer_config = cms.PSet(
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
    weightFileNames    = mvaWeightFiles
    )
# Create the VPset's for VID cuts
mvaEleID_Fall17_V2_wpLoose = configureVIDMVAEleID_V1( MVA_WPLoose )
mvaEleID_Fall17_V2_wp90 = configureVIDMVAEleID_V1( MVA_WP90, cutName="GsfEleMVAExpoScalingCut")
mvaEleID_Fall17_V2_wp80 = configureVIDMVAEleID_V1( MVA_WP80, cutName="GsfEleMVAExpoScalingCut")

mvaEleID_Fall17_V2_wpLoose.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_V2_wp90.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_V2_wp80.isPOGApproved = cms.untracked.bool(True)
