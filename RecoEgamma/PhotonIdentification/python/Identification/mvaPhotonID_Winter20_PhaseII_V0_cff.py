from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_tools import *
#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#
#
# The following MVA is derived for PhaseII samples for photons.
# See more documentation in these presentations:
# https://indico.cern.ch/event/879937/contributions/4108370/attachments/2147472/3619954/Update_PhaseII_photonIDMVA_XGBoost_TMVA_Egamma_Prasant_20112020.pdf
#

###################This ID is valid for Phase II EB only##############################3
mvaTag = "PhaseIIWinter20v0"
mvaVariablesFile = "RecoEgamma/PhotonIdentification/data/PhotonMVAEstimatorRun2VariablesFall17V1p1.txt"
mvaWeightFiles = [
    path.join(weightFileBaseDir, "PhaseII/PhotonID_MVA_barrel_Egamma_PhaseII_weight.xml"),
    path.join(weightFileBaseDir, "PhaseII/PhotonID_MVA_barrel_Egamma_PhaseII_weight.xml"), ###To avoid any crash let it be there
    ]
# Set up the VID working point parameters
wpConfig = [
            # The working point for this MVA that is expected to have about 90% signal
            # efficiency in each category for photons with pt>30 GeV (somewhat lower
            # for lower pt photons).
            {"idName" : "mvaPhoID-PhaseIIWinter20-v0-wp90",
             "cuts"   : { "EB" : 0.737502,
                          "EE" : 0.737502 }},
            # The working point for this MVA that is expected to have about 90% signal
            # efficiency in each category for photons with pt>30 GeV (somewhat lower
            # for lower pt photons).
            {"idName" : "mvaPhoID-PhaseIIWinter20-v0-wp80",
             "cuts"   : { "EB" : 0.875003,
                          "EE" : 0.875003 }},
           ]
# Create the PSet that will be fed to the MVA value map producer and the
# VPset's for VID cuts
configs = configureFullVIDMVAPhoID(mvaTag=mvaTag,
                                   variablesFile=mvaVariablesFile,
                                   weightFiles=mvaWeightFiles,
                                   wpConfig=wpConfig,
                                   # Category parameters
                                   nCategories         = cms.int32(2),
                                   categoryCuts        = category_cuts)
mvaPhoID_PhaseIIWinter20_v0_producer_config = configs["producer_config"]
mvaPhoID_PhaseIIWinter20_v0_wp90            = configs["VID_config"]["mvaPhoID-PhaseIIWinter20-v0-wp90"]
mvaPhoID_PhaseIIWinter20_v0_wp80            = configs["VID_config"]["mvaPhoID-PhaseIIWinter20-v0-wp80"]
# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to
# 1) comment out the lines below about the registry,
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#
central_id_registry.register( mvaPhoID_PhaseIIWinter20_v0_wp90.idName,
                              '5e47410687d92780e001aafcb1db208d')
central_id_registry.register( mvaPhoID_PhaseIIWinter20_v0_wp80.idName,
                              '95fa9e58694799b506a3aedcf516763e')
mvaPhoID_PhaseIIWinter20_v0_wp90.isPOGApproved = cms.bool(False)
mvaPhoID_PhaseIIWinter20_v0_wp80.isPOGApproved = cms.bool(False)
