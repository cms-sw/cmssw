import FWCore.ParameterSet.Config as cms

primitiveRPCProducer = cms.EDProducer("L1TMuonRPCTriggerPrimitivesProducer",
				Primitiverechitlabel = cms.InputTag("rpcdigis"),		
				Mapsource = cms.string('L1Trigger/L1TMuon/data/rpc/Linkboard_rpc_roll_mapping_lb_chamber2.txt'),
			        ApplyLinkBoardCut = cms.bool(True),
			        LinkBoardCut = cms.int32(2), # Number of clusters per linkboard greater than (default >2) are rejected
			        ClusterSizeCut = cms.int32(3), # Clustersize greater than (default >3) is rejected 
				maskSource = cms.string('File'),
    				maskvecfile = cms.FileInPath('RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat'),
    				deadSource = cms.string('File'),
				deadvecfile = cms.FileInPath('RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat'),
				recAlgoConfig = cms.PSet(),
				recAlgo = cms.string('RPCRecHitStandardAlgo')	
)

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify(primitiveRPCProducer, ApplyLinkBoardCut = cms.bool(False))
phase2_muon.toModify(primitiveRPCProducer, ClusterSizeCut = cms.int32(4))
