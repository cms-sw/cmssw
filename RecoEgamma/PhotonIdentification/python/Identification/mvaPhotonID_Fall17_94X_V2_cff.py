from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_tools import *
#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#
#
# The following MVA is derived for Fall17 samples for photons.
# See more documentation in these presentations:
# https://indico.cern.ch/event/697079/contributions/2968123/attachments/1632966/2604131/PhotonID_EGM_13.04.2018.pdf
#
mvaTag = "RunIIFall17v2"
mvaVariablesFile = "RecoEgamma/PhotonIdentification/data/PhotonMVAEstimatorRun2VariablesFall17V1p1.txt"
mvaWeightFiles = [
    path.join(weightFileBaseDir, "Fall17/EB_V2.weights.xml.gz"),
    path.join(weightFileBaseDir, "Fall17/EE_V2.weights.xml.gz"),
    ]
# Set up the VID working point parameters
wpConfig = [
            # The working point for this MVA that is expected to have about 90% signal
            # efficiency in each category for photons with pt>30 GeV (somewhat lower
            # for lower pt photons).
            {"idName" : "mvaPhoID-RunIIFall17-v2-wp90",
             "cuts"   : { "EB" : -0.02,
                          "EE" : -0.26 }},
            # The working point for this MVA that is expected to have about 90% signal
            # efficiency in each category for photons with pt>30 GeV (somewhat lower
            # for lower pt photons).
            {"idName" : "mvaPhoID-RunIIFall17-v2-wp80",
             "cuts"   : { "EB" : 0.42,
                          "EE" : 0.14 }},
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
mvaPhoID_RunIIFall17_v2_producer_config = configs["producer_config"]
mvaPhoID_RunIIFall17_v2_wp90            = configs["VID_config"]["mvaPhoID-RunIIFall17-v2-wp90"]
mvaPhoID_RunIIFall17_v2_wp80            = configs["VID_config"]["mvaPhoID-RunIIFall17-v2-wp80"]
# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to
# 1) comment out the lines below about the registry,
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#
central_id_registry.register( mvaPhoID_RunIIFall17_v2_wp90.idName,
                              '5c06832759b1faf7dd6fc45ed1aef3a2')
central_id_registry.register( mvaPhoID_RunIIFall17_v2_wp80.idName,
                              '3013ddce7a3ad8b54827c29f5d92282e')
mvaPhoID_RunIIFall17_v2_wp90.isPOGApproved = cms.bool(True)
mvaPhoID_RunIIFall17_v2_wp80.isPOGApproved = cms.bool(True)
