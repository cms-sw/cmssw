import FWCore.ParameterSet.Config as cms
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *
from os import path

# Egamma presentation on this ID:
# https://indico.cern.ch/event/732971/contributions/3022864/attachments/1658765/2656595/180530_egamma.pdf

mvaTag = "Fall17NoIsoV2"

weightFileDir = "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/Fall17NoIsoV2"

mvaWeightFiles = cms.vstring(
     path.join(weightFileDir, "EB1_5.weights.xml.gz"), # EB1_5
     path.join(weightFileDir, "EB2_5.weights.xml.gz"), # EB2_5
     path.join(weightFileDir, "EE_5.weights.xml.gz"), # EE_5
     path.join(weightFileDir, "EB1_10.weights.xml.gz"), # EB1_10
     path.join(weightFileDir, "EB2_10.weights.xml.gz"), # EB2_10
     path.join(weightFileDir, "EE_10.weights.xml.gz"), # EE_10
     )

categoryCuts = cms.vstring(
     "pt < 10. && abs(superCluster.eta) < 0.800", # EB1_5
     "pt < 10. && abs(superCluster.eta) >= 0.800 && abs(superCluster.eta) < 1.479", # EB2_5
     "pt < 10. && abs(superCluster.eta) >= 1.479", # EE_5
     "pt >= 10. && abs(superCluster.eta) < 0.800", # EB1_10
     "pt >= 10. && abs(superCluster.eta) >= 0.800 && abs(superCluster.eta) < 1.479", # EB2_10
     "pt >= 10. && abs(superCluster.eta) >= 1.479", # EE_10
     )

mvaEleID_Fall17_noIso_V2_wp80_container = EleMVARaw_WP(
    idName = "mvaEleID-Fall17-noIso-V2-wp80", mvaTag = mvaTag,
    cutCategory0 = "3.26449620468 - exp(-pt / 3.32657149223) * 8.84669783568", # EB1_5
    cutCategory1 = "2.83557838497 - exp(-pt / 2.15150487651) * 11.0978016567", # EB2_5
    cutCategory2 = "2.91994945177 - exp(-pt / 1.69875477522) * 24.024807824", # EE_5
    cutCategory3 = "7.1336238874 - exp(-pt / 16.5605268797) * 8.22531222391", # EB1_10
    cutCategory4 = "6.18638275782 - exp(-pt / 15.2694634284) * 7.49764565324", # EB2_10
    cutCategory5 = "5.43175865738 - exp(-pt / 15.4290075949) * 7.56899692285", # EE_10
    )

mvaEleID_Fall17_noIso_V2_wpLoose_container = EleMVARaw_WP(
    idName = "mvaEleID-Fall17-noIso-V2-wpLoose", mvaTag = mvaTag,
    cutCategory0 = "0.894411158628", # EB1_5
    cutCategory1 = "0.791966464633", # EB2_5
    cutCategory2 = "1.47104857173", # EE_5
    cutCategory3 = "-0.293962958665", # EB1_10
    cutCategory4 = "-0.250424758584", # EB2_10
    cutCategory5 = "-0.130985179031", # EE_10
    )

mvaEleID_Fall17_noIso_V2_wp90_container = EleMVARaw_WP(
    idName = "mvaEleID-Fall17-noIso-V2-wp90", mvaTag = mvaTag,
    cutCategory0 = "2.77072387339 - exp(-pt / 3.81500912145) * 8.16304860178", # EB1_5
    cutCategory1 = "1.85602317813 - exp(-pt / 2.18697654938) * 11.8568936824", # EB2_5
    cutCategory2 = "1.73489307814 - exp(-pt / 2.0163211971) * 17.013880078", # EE_5
    cutCategory3 = "5.9175992258 - exp(-pt / 13.4807294538) * 9.31966232685", # EB1_10
    cutCategory4 = "5.01598837255 - exp(-pt / 13.1280451502) * 8.79418193765", # EB2_10
    cutCategory5 = "4.16921343208 - exp(-pt / 13.2017224621) * 9.00720913211", # EE_10
    )


mvaEleID_Fall17_noIso_V2_producer_config = cms.PSet(
    mvaName             = cms.string(mvaClassName),
    mvaTag              = cms.string(mvaTag),
    nCategories         = cms.int32(6),
    categoryCuts        = categoryCuts,
    weightFileNames     = mvaWeightFiles,
    variableDefinition  = cms.string(mvaVariablesFile)
    )

mvaEleID_Fall17_noIso_V2_wp80 = configureVIDMVAEleID( mvaEleID_Fall17_noIso_V2_wp80_container )
mvaEleID_Fall17_noIso_V2_wpLoose = configureVIDMVAEleID( mvaEleID_Fall17_noIso_V2_wpLoose_container )
mvaEleID_Fall17_noIso_V2_wp90 = configureVIDMVAEleID( mvaEleID_Fall17_noIso_V2_wp90_container )

mvaEleID_Fall17_noIso_V2_wp80.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_noIso_V2_wpLoose.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_noIso_V2_wp90.isPOGApproved = cms.untracked.bool(True)
