import FWCore.ParameterSet.Config as cms

emulatorCppfDigis = cms.EDProducer("L1TMuonCPPFDigiProducer",                                   
                                   ## Input collection
                                   recHitLabel = cms.InputTag("rpcRecHits"),
                                   rpcDigiLabel = cms.InputTag("muonRPCDigis"),
                                   rpcDigiSimLinkLabel = cms.InputTag("simMuonRPCDigis", "RPCDigiSimLink"),
				   MaxClusterSize = cms.int32(3),
                                   #  cppfSource = cms.string('Geo'), #'File' for Look up table and 'Geo' for CMSSW Geometry 
                                   cppfSource = cms.string('File'), #'File' for Look up table and 'Geo' for CMSSW Geometry 
                                   
                                   cppfvecfile = cms.FileInPath('L1Trigger/L1TMuon/data/cppf/angleScale_RPC_CPPFmerged.txt')                     
                                   #    cppfvecfile = cms.FileInPath('L1Trigger/L1TMuon/data/cppf/angleScale_RPC_CPPFn1.txt')                    
                                   #    cppfvecfile = cms.FileInPath('L1Trigger/L1TMuon/data/cppf/angleScale_RPC_CPPFn2.txt')                    
                                   #    cppfvecfile = cms.FileInPath('L1Trigger/L1TMuon/data/cppf/angleScale_RPC_CPPFn3.txt')                    
                                   #    cppfvecfile = cms.FileInPath('L1Trigger/L1TMuon/data/cppf/angleScale_RPC_CPPFn4.txt')                    
                                   #    cppfvecfile = cms.FileInPath('L1Trigger/L1TMuon/data/cppf/angleScale_RPC_CPPFp1.txt')                    
                                   #    cppfvecfile = cms.FileInPath('L1Trigger/L1TMuon/data/cppf/angleScale_RPC_CPPFp2.txt')                    
                                   #    cppfvecfile = cms.FileInPath('L1Trigger/L1TMuon/data/cppf/angleScale_RPC_CPPFp3.txt')                   
                                   #    cppfvecfile = cms.FileInPath('L1Trigger/L1TMuon/data/cppf/angleScale_RPC_CPPFp4.txt')                    
                                   )

