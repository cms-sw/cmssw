import FWCore.ParameterSet.Config as cms

RPCRecHitsFilter = cms.EDFilter("RPCRecHitFilter",
                                rpcRecHitLabel = cms.untracked.string('rpcRecHits'),
                                
                                minimumNumberOfHits = cms.untracked.int32(2),
                                        
                                
                                #Use those variables for fine selection:
                                
                                #At least 1 hit in RB1in and one in RB1out of adjacent wheels and sectors
                                UseBarrel = cms.untracked.bool(False),
                                
                                
                                #At least 2 hits in 2 different rings in two adjacent sectors
                                UseEndcapPositive = cms.untracked.bool(False),
                                UseEndcapNegative = cms.untracked.bool(False),
                                
                                #Deprecated: discards the RecHhts if there is a hit in RB3 or RB4
                                CosmicsVeto = cms.untracked.bool(False)
                                )
