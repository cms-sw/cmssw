import FWCore.ParameterSet.Config as cms
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *
from os import path

# Egamma presentation on this ID:
# https://indico.cern.ch/event/732971/contributions/3022864/attachments/1658765/2656595/180530_egamma.pdf

mvaTag = "Fall17IsoV2"

weightFileDir = "RecoEgamma/ElectronIdentification/data/MVAWeightFiles/Fall17IsoV2"

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

mvaEleID_Fall17_iso_V2_wpHZZ_container = EleMVARaw_WP(
    idName = "mvaEleID-Fall17-iso-V2-wpHZZ", mvaTag = mvaTag,
    cutCategory0 = "1.26402092475", # EB1_5
    cutCategory1 = "1.17808089508", # EB2_5
    cutCategory2 = "1.33051972806", # EE_5
    cutCategory3 = "2.36464785939", # EB1_10
    cutCategory4 = "2.07880614597", # EB2_10
    cutCategory5 = "1.08080644615", # EE_10
    )

mvaEleID_Fall17_iso_V2_wp80_container = EleMVARaw_WP(
    idName = "mvaEleID-Fall17-iso-V2-wp80", mvaTag = mvaTag,
    cutCategory0 = "3.53495358797 - exp(-pt / 3.07272325141) * 9.94262764352", # EB1_5
    cutCategory1 = "3.06015605623 - exp(-pt / 1.95572234114) * 14.3091184421", # EB2_5
    cutCategory2 = "3.02052519639 - exp(-pt / 1.59784164742) * 28.719380105", # EE_5
    cutCategory3 = "7.35752275071 - exp(-pt / 15.87907864) * 7.61288809226", # EB1_10
    cutCategory4 = "6.41811074032 - exp(-pt / 14.730562874) * 6.96387331587", # EB2_10
    cutCategory5 = "5.64936312428 - exp(-pt / 16.3664949747) * 7.19607610311", # EE_10
    )

mvaEleID_Fall17_iso_V2_wpLoose_container = EleMVARaw_WP(
    idName = "mvaEleID-Fall17-iso-V2-wpLoose", mvaTag = mvaTag,
    cutCategory0 = "0.700642584415", # EB1_5
    cutCategory1 = "0.739335420875", # EB2_5
    cutCategory2 = "1.45390456109", # EE_5
    cutCategory3 = "-0.146270871164", # EB1_10
    cutCategory4 = "-0.0315850882679", # EB2_10
    cutCategory5 = "-0.0321841194737", # EE_10
    )

mvaEleID_Fall17_iso_V2_wp90_container = EleMVARaw_WP(
    idName = "mvaEleID-Fall17-iso-V2-wp90", mvaTag = mvaTag,
    cutCategory0 = "2.84704783417 - exp(-pt / 3.32529515837) * 9.38050947827", # EB1_5
    cutCategory1 = "2.03833922005 - exp(-pt / 1.93288758682) * 15.364588247", # EB2_5
    cutCategory2 = "1.82704158461 - exp(-pt / 1.89796754399) * 19.1236071158", # EE_5
    cutCategory3 = "6.12931925263 - exp(-pt / 13.281753835) * 8.71138432196", # EB1_10
    cutCategory4 = "5.26289004857 - exp(-pt / 13.2154971491) * 8.0997882835", # EB2_10
    cutCategory5 = "4.37338792902 - exp(-pt / 14.0776094696) * 8.48513324496", # EE_10
    )


mvaEleID_Fall17_iso_V2_producer_config = cms.PSet(
    mvaName             = cms.string(mvaClassName),
    mvaTag              = cms.string(mvaTag),
    nCategories         = cms.int32(6),
    categoryCuts        = categoryCuts,
    weightFileNames     = mvaWeightFiles,
    variableDefinition  = cms.string(mvaVariablesFile)
    )

mvaEleID_Fall17_iso_V2_wpHZZ = configureVIDMVAEleID( mvaEleID_Fall17_iso_V2_wpHZZ_container )
mvaEleID_Fall17_iso_V2_wp80 = configureVIDMVAEleID( mvaEleID_Fall17_iso_V2_wp80_container )
mvaEleID_Fall17_iso_V2_wpLoose = configureVIDMVAEleID( mvaEleID_Fall17_iso_V2_wpLoose_container )
mvaEleID_Fall17_iso_V2_wp90 = configureVIDMVAEleID( mvaEleID_Fall17_iso_V2_wp90_container )

mvaEleID_Fall17_iso_V2_wpHZZ.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_iso_V2_wp80.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_iso_V2_wpLoose.isPOGApproved = cms.untracked.bool(True)
mvaEleID_Fall17_iso_V2_wp90.isPOGApproved = cms.untracked.bool(True)
