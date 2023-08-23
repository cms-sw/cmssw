import FWCore.ParameterSet.Config as cms
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *
from os import path

#Egamma presentation on this ID for Run3:
#https://indico.cern.ch/event/1220628/contributions/5134878/attachments/2546114/4384580/Run%203%20Electron%20MVA%20based%20ID%20training.pdf

mvaTag = "RunIIIWinter22NoIsoV1"

weightFileDir = "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/Winter22NoIsoV1"


mvaWeightFiles = cms.vstring(
     path.join(weightFileDir, "EB1_5.weights.root"), # EB1_5
     path.join(weightFileDir, "EB2_5.weights.root"), # EB2_5
     path.join(weightFileDir, "EE_5.weights.root"), # EE_5
     path.join(weightFileDir, "EB1_10.weights.root"), # EB1_10
     path.join(weightFileDir, "EB2_10.weights.root"), # EB2_10
     path.join(weightFileDir, "EE_10.weights.root"), # EE_10
     )

mvaEleID_RunIIIWinter22_noIso_V1_wp80_container = EleMVA_WP(
    idName = "mvaEleID-RunIIIWinter22-noIso-V1-wp80", mvaTag = mvaTag,
    cutCategory0 = "0.99776583820026143", # EB1_5
    cutCategory1 = "0.99399710641666705", # EB2_5
    cutCategory2 = "0.89762967983679642", # EE_5
    cutCategory3 = "0.99997133733482069", # EB1_10
    cutCategory4 = "0.99991566148661426", # EB2_10
    cutCategory5 = "0.99712023523348758", # EE_10
    )



mvaEleID_RunIIIWinter22_noIso_V1_wp90_container = EleMVA_WP(
    idName = "mvaEleID-RunIIIWinter22-noIso-V1-wp90", mvaTag = mvaTag,
    cutCategory0 = "0.9870981346957135", # EB1_5
    cutCategory1 = "0.95756807831082225", # EB2_5
    cutCategory2 = "0.4195020250389494", # EE_5
    cutCategory3 = "0.99981763428587134", # EB1_10
    cutCategory4 = "0.99936974968805936", # EB2_10
    cutCategory5 = "0.96553633326857091", # EE_10
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
