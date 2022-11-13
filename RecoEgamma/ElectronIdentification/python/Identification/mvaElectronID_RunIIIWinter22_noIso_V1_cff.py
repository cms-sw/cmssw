import FWCore.ParameterSet.Config as cms
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *
from os import path

# Egamma presentation on this ID for Run3:
#https://indico.cern.ch/event/1220628/contributions/5134878/attachments/2546114/4384580/Run%203%20Electron%20MVA%20based%20ID%20training.pdf

mvaTag = "RunIIIWinter22NoIsoV1"

#weightFileDir = "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/RunIIIWinter22NoIsoV1"
weightFileDir = "/afs/cern.ch/work/p/prrout/public/pkltoxml/Non_Iso"


mvaWeightFiles = cms.vstring(
     path.join(weightFileDir, "EB1_5.weights.xml"), # EB1_5
     path.join(weightFileDir, "EB2_5.weights.xml"), # EB2_5
     path.join(weightFileDir, "EE_5.weights.xml"), # EE_5
     path.join(weightFileDir, "EB1_10.weights.xml"), # EB1_10
     path.join(weightFileDir, "EB2_10.weights.xml"), # EB2_10
     path.join(weightFileDir, "EE_10.weights.xml"), # EE_10
     )

mvaEleID_RunIIIWinter22_noIso_V1_wp80_container = EleMVARaw_WP(
    idName = "mvaEleID-RunIIIWinter22-noIso-V1-wp80", mvaTag = mvaTag,
    cutCategory0 = " 0.871070015 ", # EB1_5
    cutCategory1 = " 0.787631679 ", # EB2_5
    cutCategory2 = " 0.72063427 ", # EE_5
    cutCategory3 = " 0.984803283 ", # EB1_10
    cutCategory4 = " 0.968268013 ", # EB2_10
    cutCategory5 = " 0.937130225 ", # EE_10
    )



mvaEleID_RunIIIWinter22_noIso_V1_wp90_container = EleMVARaw_WP(
    idName = "mvaEleID-RunIIIWinter22-noIso-V1-wp90", mvaTag = mvaTag,
    cutCategory0 = " 0.720341086 ", # EB1_5
    cutCategory1 = " 0.601036561 ", # EB2_5
    cutCategory2 = " 0.492077446 ", # EE_5
    cutCategory3 = " 0.953999406 ", # EB1_10
    cutCategory4 = " 0.90531317 ", # EB2_10
    cutCategory5 = " 0.81990177 ", # EE_10
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
