import FWCore.ParameterSet.Config as cms

from Geometry.MuonNumbering.muonOffsetESProducer_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(muonOffsetESProducer,
                fromDD4Hep = cms.bool(True)
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common

phase2_common.toModify(muonOffsetESProducer,
                       names = cms.vstring(
                           "MuonCommonNumbering", 
                           "MuonBarrel", 
                           "MuonEndcap",
                           "MuonBarrelWheels", 
                           "MuonBarrelStation1", 
                           "MuonBarrelStation2", 
                           "MuonBarrelStation3", 
                           "MuonBarrelStation4", 
                           "MuonBarrelSuperLayer", 
                           "MuonBarrelLayer", 
                           "MuonBarrelWire", 
                           "MuonRpcPlane1I", 
                           "MuonRpcPlane1O", 
                           "MuonRpcPlane2I", 
                           "MuonRpcPlane2O", 
                           "MuonRpcPlane3S", 
                           "MuonRpcPlane4", 
                           "MuonRpcChamberLeft", 
                           "MuonRpcChamberMiddle", 
                           "MuonRpcChamberRight", 
                           "MuonRpcEndcap1", 
                           "MuonRpcEndcap2", 
                           "MuonRpcEndcap3", 
                           "MuonRpcEndcap4", 
                           "MuonRpcEndcapSector", 
                           "MuonRpcEndcapChamberB1", 
                           "MuonRpcEndcapChamberB2", 
                           "MuonRpcEndcapChamberB3", 
                           "MuonRpcEndcapChamberC1", 
                           "MuonRpcEndcapChamberC2", 
                           "MuonRpcEndcapChamberC3", 
                           "MuonRpcEndcapChamberE1", 
                           "MuonRpcEndcapChamberE2", 
                           "MuonRpcEndcapChamberE3", 
                           "MuonRpcEndcapChamberF1", 
                           "MuonRpcEndcapChamberF2", 
                           "MuonRpcEndcapChamberF3", 
                           "MuonRpcEndcapChamberG1", 
                           "MuonRpcEndcapChamberH1", 
                           "MuonEndcapStation1", 
                           "MuonEndcapStation2", 
                           "MuonEndcapStation3", 
                           "MuonEndcapStation4", 
                           "MuonEndcapSubrings", 
                           "MuonEndcapSectors", 
                           "MuonEndcapLayers", 
                           "MuonEndcapRing1", 
                           "MuonEndcapRing2", 
                           "MuonEndcapRing3", 
                           "MuonEndcapRingA", 
                           "MuonGEMEndcap", 
                           "MuonGEMEndcap2", 
                           "MuonGEMSector", 
                           "MuonGEMChamber", 
                           "MuonGE0Sector", 
                           "MuonGE0Layer", 
                           "MuonGE0Chamber")
                   )
