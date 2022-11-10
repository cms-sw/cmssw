import FWCore.ParameterSet.Config as cms
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *
from os import path

# Egamma presentation on this ID
# 

mvaTag = "RunIIIWinter22IsoV1"

weightFileDir = "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/RunIIIWinter22"

mvaWeightFiles = [
     path.join(weightFileDir, "EB1_5.weights.root"), # EB1_5
     path.join(weightFileDir, "EB2_5.weights.root"), # EB2_5
     path.join(weightFileDir, "EE_5.weights.root"), # EE_5
     path.join(weightFileDir, "EB1_10.weights.root"), # EB1_10
     path.join(weightFileDir, "EB2_10.weights.root"), # EB2_10
     path.join(weightFileDir, "EE_10.weights.root"), # EE_10
     ]

mvaEleID_RunIIIWinter22_iso_V1_wp80_container = EleMVARaw_WP(
    idName = "mvaEleID-RunIIIWinter22-iso-V1-wp80", mvaTag = mvaTag,
    cutCategory0 = "0.956654715538025", # EB1_5
    cutCategory1 = "0.9244146823883057", # EB2_5
    cutCategory2 = "0.8489419341087341", # EE_5
    cutCategory3 = "0.9934913158416748", # EB1_10
    cutCategory4 = "0.9874098300933838", # EB2_10
    cutCategory5 = "0.9670001745223998", # EE_10
    )


mvaEleID_RunIIIWinter22_iso_V1_wp90_container = EleMVARaw_WP(
    idName = "mvaEleID-RunIIIWinter22-iso-V1-wp90", mvaTag = mvaTag,
    cutCategory0 = "0.9001831412315369", # EB1_5
    cutCategory1 = "0.8278245985507964", # EB2_5
    cutCategory2 = "0.672588300704956", # EE_5
    cutCategory3 = "0.9818042814731598", # EB1_10
    cutCategory4 = "0.9656652748584746", # EB2_10
    cutCategory5 = "0.9055343508720398", # EE_10
    )


workingPoints = dict(
    wp80 = mvaEleID_RunIIIWinter22_iso_V1_wp80_container,
    wp90 = mvaEleID_RunIIIWinter22_iso_V1_wp90_container
)

mvaEleID_RunIIIWinter22_iso_V1_producer_config = cms.PSet(
    mvaName             = cms.string(mvaClassName),
    mvaTag              = cms.string(mvaTag),
    nCategories         = cms.int32(6),
    categoryCuts        = cms.vstring(*EleMVA_6CategoriesCuts),
    weightFileNames     = cms.vstring(*mvaWeightFiles),
    variableDefinition  = cms.string(mvaVariablesFileRun3)
    )


mvaEleID_RunIIIWinter22_iso_V1_wp80 = configureVIDMVAEleID( mvaEleID_RunIIIWinter22_iso_V1_wp80_container )
mvaEleID_RunIIIWinter22_iso_V1_wp90 = configureVIDMVAEleID( mvaEleID_RunIIIWinter22_iso_V1_wp90_container )


mvaEleID_RunIIIWinter22_iso_V1_wp80.isPOGApproved = cms.untracked.bool(True)
mvaEleID_RunIIIWinter22_iso_V1_wp90.isPOGApproved = cms.untracked.bool(True)
