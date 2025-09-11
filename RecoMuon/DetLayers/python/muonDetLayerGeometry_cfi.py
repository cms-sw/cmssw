import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the muon DetLayers.
#

MuonDetLayerGeometryESProducer = cms.ESProducer("MuonDetLayerGeometryESProducer", useUpdatedRPCIsFront=False)

from Configuration.Eras.Modifier_run3_RPC_2025_cff import run3_RPC_2025
from Configuration.Eras.Modifier_phase2_RPC_cff import phase2_RPC

run3_RPC_2025.toModify(MuonDetLayerGeometryESProducer, useUpdatedRPCIsFront=True)
phase2_RPC.toModify(MuonDetLayerGeometryESProducer, useUpdatedRPCIsFront=True)
