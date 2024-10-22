import FWCore.ParameterSet.Config as cms
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *
from os import path

mvaTag = "Summer18ULIdIso"

weightFileDir = "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/Summer_18UL_ID_ISO"

mvaWeightFiles = cms.vstring(
     path.join(weightFileDir, "EB1_5.weights.root"), # EB1_5
     path.join(weightFileDir, "EB2_5.weights.root"), # EB2_5
     path.join(weightFileDir, "EE_5.weights.root"), # EE_5
     path.join(weightFileDir, "EB1_10.weights.root"), # EB1_10
     path.join(weightFileDir, "EB2_10.weights.root"), # EB2_10
     path.join(weightFileDir, "EE_10.weights.root"), # EE_10
     )

categoryCuts = cms.vstring(
     "pt < 10. & abs(superCluster.eta) < 0.800", # EB1_5
     "pt < 10. & abs(superCluster.eta) >= 0.800 & abs(superCluster.eta) < 1.479", # EB2_5
     "pt < 10. & abs(superCluster.eta) >= 1.479", # EE_5
     "pt >= 10. & abs(superCluster.eta) < 0.800", # EB1_10
     "pt >= 10. & abs(superCluster.eta) >= 0.800 & abs(superCluster.eta) < 1.479", # EB2_10
     "pt >= 10. & abs(superCluster.eta) >= 1.479", # EE_10
     )

mvaEleID_Summer18UL_ID_ISO_HZZ_container = EleMVARaw_WP(
    idName = "mvaEleID-Summer18UL-ID-ISO-HZZ", mvaTag = mvaTag,
    cutCategory0 = "1.49603193295", # EB1_5
    cutCategory1 = "1.52414154008", # EB2_5
    cutCategory2 = "1.77694249574", # EE_5
    cutCategory3 = "0.199463934736", # EB1_10
    cutCategory4 = "0.076063564084", # EB2_10
    cutCategory5 = "-0.572118857519", # EE_10
    )


mvaEleID_Summer18UL_ID_ISO_producer_config = cms.PSet(
    mvaName             = cms.string(mvaClassName),
    mvaTag              = cms.string(mvaTag),
    nCategories         = cms.int32(6),
    categoryCuts        = categoryCuts,
    weightFileNames     = mvaWeightFiles,
    variableDefinition  = cms.string(mvaVariablesFile)
    )

mvaEleID_Summer18UL_ID_ISO_HZZ = configureVIDMVAEleID( mvaEleID_Summer18UL_ID_ISO_HZZ_container )

mvaEleID_Summer18UL_ID_ISO_HZZ.isPOGApproved = cms.untracked.bool(True)
