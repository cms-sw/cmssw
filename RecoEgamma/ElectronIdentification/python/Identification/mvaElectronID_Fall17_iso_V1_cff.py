import FWCore.ParameterSet.Config as cms

# Documentation of the MVA
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/MultivariateElectronIdentificationRun2
# https://rembserj.web.cern.ch/rembserj/notes/Electron_MVA_ID_2017_documentation

#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#

# This MVA implementation class name
mvaFall17ClassName = "ElectronMVAEstimatorRun2Fall17Iso"
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
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB1_5_2017_puinfo_iso_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB2_5_2017_puinfo_iso_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EE_5_2017_puinfo_iso_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB1_10_2017_puinfo_iso_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EB2_10_2017_puinfo_iso_BDT.weights.xml.gz",
    "RecoEgamma/ElectronIdentification/data/Fall17/EIDmva_EE_10_2017_puinfo_iso_BDT.weights.xml.gz"
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
idName90 = "mvaEleID-Fall17-iso-V1-wp90"
MVA_WP90 = EleMVA_WP(
    idName = idName90,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0_C0 = 0.9387070396095831, # EB1 low pt
    cutCategory0_C1 = 2.6525585228167636,
    cutCategory0_C2 = 0.8222647164151365,
    cutCategory1_C0 = 0.8948802925677235, # EB2 low pt
    cutCategory1_C1 = 2.7645670358783523,
    cutCategory1_C2 = 0.4123381218697539,
    cutCategory2_C0 = -1830.8583661119892, # EE low pt
    cutCategory2_C1 = -36578.11055382301,
    cutCategory2_C2 = -1831.2083578116517,
    cutCategory3_C0 = 0.9717674837607253, # EB1
    cutCategory3_C1 = 8.912850985100356,
    cutCategory3_C2 = 1.9712414940437244,
    cutCategory4_C0 = 0.9458745023265976, # EB2
    cutCategory4_C1 = 8.83104420392795,
    cutCategory4_C2 = 2.40849932040698,
    cutCategory5_C0 = 0.8979112012086751, # EE
    cutCategory5_C1 = 9.814082144168015,
    cutCategory5_C2 = 4.171581694893849
)

idName80 = "mvaEleID-Fall17-iso-V1-wp80"
MVA_WP80 = EleMVA_WP(
    idName = idName80,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0_C0 = 0.9725509559754997, # EB1 low pt
    cutCategory0_C1 = 2.976593261509491,
    cutCategory0_C2 = 0.2653858736397496,
    cutCategory1_C0 = 0.9508038141601247, # EB2 low pt
    cutCategory1_C1 = 2.6633500558725713,
    cutCategory1_C2 = 0.2355820499260076,
    cutCategory2_C0 = 0.9365037167596238, # EE low pt
    cutCategory2_C1 = 1.5765442323949856,
    cutCategory2_C2 = 3.067015289215309,
    cutCategory3_C0 = 0.9896562087723659, # EB1
    cutCategory3_C1 = 10.342490511998674,
    cutCategory3_C2 = 0.40204156417414094,
    cutCategory4_C0 = 0.9819232656533827, # EB2
    cutCategory4_C1 = 9.05548836482051,
    cutCategory4_C2 = 0.772674931169389,
    cutCategory5_C0 = 0.9625098201744635, # EE
    cutCategory5_C1 = 8.42589315557279,
    cutCategory5_C2 = 2.2916152615134173
)

### WP tuned for HZZ analysis with very high efficiency (about 98%)
# The working points were found by requiring the same signal efficiencies in
# each category as for the Spring 16 HZZ ID
# (see RecoEgamma/ElectronIdentification/python/Identification/mvaElectronID_Spring16_HZZ_V1_cff.py)
idNamewpLoose = "mvaEleID-Fall17-iso-V1-wpLoose"
MVA_WPLoose = EleMVA_WP(
    idName = idNamewpLoose,
    mvaValueMapName = mvaValueMapName,           # map with MVA values for all particles
    mvaCategoriesMapName = mvaCategoriesMapName, # map with category index for all particles
    cutCategory0 =  -0.09564086146419018, # EB1 low pt
    cutCategory1 =  -0.28229916981926795, # EB2 low pt
    cutCategory2 =  -0.05466682296962322, # EE low pt
    cutCategory3 =  -0.833466688584422  , # EB1
    cutCategory4 =  -0.7677000247570116 , # EB2
    cutCategory5 =  -0.6917305995653829   # EE
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
                 ("ele_pfPhotonIso"                   ,  None, None), #
                 ("ele_pfChargedHadIso"               ,  None, None), # PF isolations
                 ("ele_pfNeutralHadIso"               ,  None, None), #
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
mvaEleID_Fall17_iso_V1_producer_config = cms.PSet(
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
