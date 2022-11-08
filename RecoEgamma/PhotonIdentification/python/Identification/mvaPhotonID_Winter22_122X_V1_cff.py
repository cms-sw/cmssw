from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_tools import *
#
# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID
#
#
#The following MVA is derived for Winter22 Gjet samples in CMSSW_12_2_1 campaign for photons.
#See more documentation in these presentations:
#https://indico.cern.ch/event/1182858/#2-update-on-run3-photon-mva-id
#https://indico.cern.ch/event/1188686/#2-update-on-run3-photon-mva-id
#https://indico.cern.ch/event/1192144/#2-update-on-run3-photon-mva-id
#https://indico.cern.ch/event/1192146/#2-update-on-run3-photon-mva-id
#https://indico.cern.ch/event/1192145/#2-update-on-run3-photon-mva-id
#https://indico.cern.ch/event/1204272/#2-update-on-run3-photon-mva-id
#
#
 
mvaTag = "RunIIIWinter22v1"
mvaVariablesFile = "RecoEgamma/PhotonIdentification/data/PhotonMVAEstimatorRun3VariablesWinter22V1.txt"
mvaWeightFiles = [
    path.join(weightFileBaseDir, "RunIII_Winter22/PhoMVA_ID_EB_V1.weights.root"),
    path.join(weightFileBaseDir, "RunIII_Winter22/PhoMVA_ID_EE_V1.weights.root"),
    ]
# Set up the VID working point parameters
wpConfig = [
            # The working point for this MVA that is expected to have about 90% signal
            # efficiency in each category for photons with pt>10 GeV.
            {"idName" : "mvaPhoID-RunIIIWinter22-v1-wp90",
             "cuts"   : { "EB" : 0.0439603,
                          "EE" : -0.249526 }},
            # The working point for this MVA that is expected to have about 90% signal
            # efficiency in each category for photons with pt>10 GeV.
            {"idName" : "mvaPhoID-RunIIIWinter22-v1-wp80",
             "cuts"   : { "EB" : 0.420473,
                          "EE" : 0.203451 }},
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
mvaPhoID_RunIIIWinter22_v1_producer_config = configs["producer_config"]
mvaPhoID_RunIIIWinter22_v1_wp90            = configs["VID_config"]["mvaPhoID-RunIIIWinter22-v1-wp90"]
mvaPhoID_RunIIIWinter22_v1_wp80            = configs["VID_config"]["mvaPhoID-RunIIIWinter22-v1-wp80"]
# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to
# 1) comment out the lines below about the registry,
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.
#

central_id_registry.register( mvaPhoID_RunIIIWinter22_v1_wp90.idName,
                              '2720b451f89dd72162f4a1de626a03ee098c8352')
central_id_registry.register( mvaPhoID_RunIIIWinter22_v1_wp80.idName,
                              'c198ffac6a62f5b64b1db5190048903722d29a66')
mvaPhoID_RunIIIWinter22_v1_wp90.isPOGApproved = cms.bool(True)
mvaPhoID_RunIIIWinter22_v1_wp80.isPOGApproved = cms.bool(True)
