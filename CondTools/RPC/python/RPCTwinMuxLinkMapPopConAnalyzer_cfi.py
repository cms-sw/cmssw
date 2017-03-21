import FWCore.ParameterSet.Config as cms

RPCTwinMuxLinkMapPopConAnalyzer = cms.EDAnalyzer('RPCTwinMuxLinkMapPopConAnalyzer'
                                                 , record = cms.string('RPCTwinMuxLinkMapRcd')
                                                 , Source = cms.PSet(
                                                     identifier = cms.string('RPCTwinMuxLinkMapHandler')
                                                     , dataTag = cms.string('RPCTwinMuxLinkMap_v1')
                                                     , sinceRun = cms.uint64(1)
                                                     # File provided by K. Bunkowski
                                                     , inputFile = cms.FileInPath('CondTools/RPC/data/RPCTwinMuxLinkMapInput.txt')
                                                     , wheelFED = cms.vint32(1395, 1391, 1390, 1393, 1394)
                                                     , wheelSectorAMC = cms.vint64(0x123456789ABC, 0x123456789ABC, 0x123456789ABC, 0x123456789ABC, 0x123456789ABC)
                                                     , txtFile = cms.untracked.string('')
                                                 )
)
