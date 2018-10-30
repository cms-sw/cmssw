from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_tools import *

# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID

# The following MVA is derived for Fall17 samples for non-triggering photons.
# See more documentation in these presentations:
# https://indico.cern.ch/event/662751/contributions/2778043/attachments/1562017/2459674/EGamma_WorkShop_21.11.17_Debabrata.pdf

mvaTag                       = "RunIIFall17v1"
mvaVariablesFile             = "RecoEgamma/PhotonIdentification/data/PhotonMVAEstimatorRun2VariablesFall17.txt"
mvaWeightFiles = [
    path.join(weightFileBaseDir, "Fall17/EB_V1.weights.xml.gz"),
    path.join(weightFileBaseDir, "Fall17/EE_V1.weights.xml.gz"),
    ]

# Set up the VID working point parameters
wpConfig = [
            # The working point for this MVA that is expected to have about 90% signal
            # efficiency in each category for photons with pt>30 GeV (somewhat lower
            # for lower pt photons).
            {"idName" : "mvaPhoID-RunIIFall17-v1-wp90",
             "cuts"   : { "EB" : 0.27,
                          "EE" : 0.14 }},
            # The working point for this MVA that is expected to have about 90% signal
            # efficiency in each category for photons with pt>30 GeV (somewhat lower
            # for lower pt photons).
            {"idName" : "mvaPhoID-RunIIFall17-v1-wp80",
             "cuts"   : { "EB" : 0.67,
                          "EE" : 0.54 }},
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
mvaPhoID_RunIIFall17_v1_producer_config = configs["producer_config"]
mvaPhoID_RunIIFall17_v1_wp90            = configs["VID_config"]["mvaPhoID-RunIIFall17-v1-wp90"]
mvaPhoID_RunIIFall17_v1_wp80            = configs["VID_config"]["mvaPhoID-RunIIFall17-v1-wp80"]

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to
# 1) comment out the lines below about the registry,
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.

central_id_registry.register( mvaPhoID_RunIIFall17_v1_wp90.idName,
                              '834dd792692b6a62786bd1caa6b53a68')
central_id_registry.register( mvaPhoID_RunIIFall17_v1_wp80.idName,
                              '7e48c47329d7d1eb889100ed03a02ba9')

mvaPhoID_RunIIFall17_v1_wp90.isPOGApproved = cms.untracked.bool(True)
mvaPhoID_RunIIFall17_v1_wp80.isPOGApproved = cms.untracked.bool(True)
