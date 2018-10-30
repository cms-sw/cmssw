from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_tools import *

# In this file we define the locations of the MVA weights, cuts on the MVA values
# for specific working points, and configure those cuts in VID

# The following MVA is derived for Spring16 MC samples for non-triggering photons.
# See more documentation in these presentations:
#    https://indico.cern.ch/event/491509/contributions/2226579/attachments/1303047/1946168/EGamma_PhoID_Update_IKucher.pdf
#    https://indico.cern.ch/event/578399/contributions/2344916/attachments/1357735/2053119/EGamma_PhoID_Update_19_10_2016.pdf

mvaTag           = "Run2Spring16NonTrigV1"
mvaVariablesFile = "RecoEgamma/PhotonIdentification/data/PhotonMVAEstimatorRun2VariablesSpring16.txt"
mvaWeightFiles = [
    path.join(weightFileBaseDir, "Spring16/EB_V1.weights.xml.gz"),
    path.join(weightFileBaseDir, "Spring16/EE_V1.weights.xml.gz"),
    ]
effAreasPath_pho = "RecoEgamma/PhotonIdentification/data/Spring16/effAreaPhotons_cone03_pfPhotons_90percentBased_3bins.txt"

# Set up the VID working point parameters
wpConfig = [
            # The working point for this MVA that is expected to have about 90% signal
            # efficiency in each category for photons with pt>30 GeV (somewhat lower
            # for lower pt photons).
            {"idName" : "mvaPhoID-Spring16-nonTrig-V1-wp90",
             "cuts"   : { "EB" : 0.2,
                          "EE" : 0.2 }},
            # The working point for this MVA that is expected to have about 90% signal
            # efficiency in each category for photons with pt>30 GeV (somewhat lower
            # for lower pt photons).
            {"idName" : "mvaPhoID-Spring16-nonTrig-V1-wp80",
             "cuts"   : { "EB" : 0.68,
                          "EE" : 0.60 }},
           ]

# Create the PSet that will be fed to the MVA value map producer and the
# VPset's for VID cuts
configs = configureFullVIDMVAPhoID(mvaTag=mvaTag,
                                   variablesFile=mvaVariablesFile,
                                   weightFiles=mvaWeightFiles,
                                   wpConfig=wpConfig,
    # Category parameters
    nCategories         = cms.int32(2),
    categoryCuts        = category_cuts,
    # In this MVA for endcap the corrected photon isolation is defined as
    # iso = max( photon_isolation_raw - rho*effArea - coeff*pt, cutoff)
    # as discussed in the indico presentations listed in the beginning of this file.
    effAreasConfigFile = cms.FileInPath(effAreasPath_pho),
    # The coefficients "coeff" for the formula above for linear pt scaling correction
    # the first value is for EB, the second is for EE
    # NOTE: even though the EB coefficient is provided, it is not presently used in the MVA.
    # For EB, the uncorrected raw photon isolation is used instead.
    phoIsoPtScalingCoeff = cms.vdouble(0.0053,0.0034),
    # The cutoff for the formula above
    phoIsoCutoff = cms.double(2.5))

mvaPhoID_Spring16_nonTrig_V1_producer_config = configs["producer_config"]
mvaPhoID_Spring16_nonTrig_V1_wp90 = configs["VID_config"]["mvaPhoID-Spring16-nonTrig-V1-wp90"]
mvaPhoID_Spring16_nonTrig_V1_wp80 = configs["VID_config"]["mvaPhoID-Spring16-nonTrig-V1-wp80"]

# The MD5 sum numbers below reflect the exact set of cut variables
# and values above. If anything changes, one has to
# 1) comment out the lines below about the registry,
# 2) run "calculateIdMD5 <this file name> <one of the VID config names just above>
# 3) update the MD5 sum strings below and uncomment the lines again.

central_id_registry.register( mvaPhoID_Spring16_nonTrig_V1_wp90.idName,
                              '36efe663348f95de0bc1cfa8dc7fa8fe')
central_id_registry.register( mvaPhoID_Spring16_nonTrig_V1_wp80.idName,
                              'beb95233f7d1e033ad9e20cf3d804ba0')

mvaPhoID_Spring16_nonTrig_V1_wp90.isPOGApproved = cms.untracked.bool(True)
mvaPhoID_Spring16_nonTrig_V1_wp80.isPOGApproved = cms.untracked.bool(True)
