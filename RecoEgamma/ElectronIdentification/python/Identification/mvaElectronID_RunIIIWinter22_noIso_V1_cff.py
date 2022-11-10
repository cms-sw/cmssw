import FWCore.ParameterSet.Config as cms
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *
from os import path

# Egamma presentation on this ID
#

mvaTag = "RunIIIWinter22NoIsoV1"

weightFileDir = "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/RunIIIWinter22"

mvaWeightFiles = cms.vstring(
     path.join(weightFileDir, "EB1_5.weights.root"), # EB1_5
     path.join(weightFileDir, "EB2_5.weights.root"), # EB2_5
     path.join(weightFileDir, "EE_5.weights.root"), # EE_5
     path.join(weightFileDir, "EB1_10.weights.root"), # EB1_10
     path.join(weightFileDir, "EB2_10.weights.root"), # EB2_10
     path.join(weightFileDir, "EE_10.weights.root"), # EE_10
     )

mvaEleID_RunIIIWinter22_noIso_V1_wp80_container = EleMVARaw_WP(
    idName = "mvaEleID-RunIIIWinter22-noIso-V1-wp80", mvaTag = mvaTag,
    cutCategory0 = "0.8710429549217223", # EB1_5
    cutCategory1 = "0.791666269302368", # EB2_5
    cutCategory2 = "0.7300988435745238", # EE_5
    cutCategory3 = "0.9846112370491028", # EB1_10
    cutCategory4 = "0.9685800075531006", # EB2_10
    cutCategory5 = "0.9377422094345091", # EE_10
    )



mvaEleID_RunIIIWinter22_noIso_V1_wp90_container = EleMVARaw_WP(
    idName = "mvaEleID-RunIIIWinter22-noIso-V1-wp90", mvaTag = mvaTag,
    cutCategory0 = "0.7206249237060545", # EB1_5
    cutCategory1 = "0.6127320051193237", # EB2_5
    cutCategory2 = "0.5144159138202666", # EE_5
    cutCategory3 = "0.9536022126674653", # EB1_10
    cutCategory4 = "0.905023992061615", # EB2_10
    cutCategory5 = "0.8227859497070312", # EE_10
    )

workingPoints = dict(
    wp80 = mvaEleID_RunIIIWinter22_noIso_V1_wp80_container,
    wp90 = mvaEleID_RunIIIWinter22_noIso_V1_wp90_container
)

mvaEleID_RunIIIWinter22_noIso_V1_producer_config = cms.PSet(
    mvaName             = cms.string(mvaClassName),
    mvaTag              = cms.string(mvaTag),
    nCategories         = cms.int32(6),
    categoryCuts        = cms.vstring(*EleMVA_6CategoriesCuts),
    weightFileNames     = mvaWeightFiles,
    variableDefinition  = cms.string(mvaVariablesFileRun3)
    )

mvaEleID_RunIIIWinter22_noIso_V1_wp80 = configureVIDMVAEleID( mvaEleID_RunIIIWinter22_noIso_V1_wp80_container )
mvaEleID_RunIIIWinter22_noIso_V1_wp90 = configureVIDMVAEleID( mvaEleID_RunIIIWinter22_noIso_V1_wp90_container )

mvaEleID_RunIIIWinter22_noIso_V1_wp80.isPOGApproved = cms.untracked.bool(True)
mvaEleID_RunIIIWinter22_noIso_V1_wp90.isPOGApproved = cms.untracked.bool(True)
