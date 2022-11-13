import FWCore.ParameterSet.Config as cms
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *
from os import path

# Egamma presentation on this ID for Run3
# https://indico.cern.ch/event/1220628/contributions/5134878/attachments/2546114/4384580/Run%203%20Electron%20MVA%20based%20ID%20training.pdf

mvaTag = "RunIIIWinter22IsoV1"

#weightFileDir = "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/RunIIIWinter22IsoV1"
weightFileDir = "/afs/cern.ch/work/p/prrout/public/pkltoxml/Iso_Clusterbased"

mvaWeightFiles = [
     path.join(weightFileDir, "EB1_5.weights.xml"), # EB1_5
     path.join(weightFileDir, "EB2_5.weights.xml"), # EB2_5
     path.join(weightFileDir, "EE_5.weights.xml"), # EE_5
     path.join(weightFileDir, "EB1_10.weights.xml"), # EB1_10
     path.join(weightFileDir, "EB2_10.weights.xml"), # EB2_10
     path.join(weightFileDir, "EE_10.weights.xml"), # EE_10
     ]

mvaEleID_RunIIIWinter22_iso_V1_wp80_container = EleMVARaw_WP(
    idName = "mvaEleID-RunIIIWinter22-iso-V1-wp80", mvaTag = mvaTag,
    cutCategory0 = "0.956646311", # EB1_5
    cutCategory1 = "0.924156857 ", # EB2_5
    cutCategory2 = "0.849070287 ", # EE_5
    cutCategory3 = "0.993499804", # EB1_10
    cutCategory4 = "0.987464821 ", # EB2_10
    cutCategory5 = "0.966739392 ", # EE_10
    )


mvaEleID_RunIIIWinter22_iso_V1_wp90_container = EleMVARaw_WP(
    idName = "mvaEleID-RunIIIWinter22-iso-V1-wp90", mvaTag = mvaTag,
    cutCategory0 = " 0.900200486 ", # EB1_5
    cutCategory1 = " 0.824187922 ", # EB2_5
    cutCategory2 = " 0.672394681 ", # EE_5
    cutCategory3 = " 0.981543684 ", # EB1_10
    cutCategory4 = " 0.96563136 ", # EB2_10
    cutCategory5 = " 0.904820812 ", # EE_10
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
