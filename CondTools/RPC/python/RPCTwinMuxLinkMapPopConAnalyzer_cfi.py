import FWCore.ParameterSet.Config as cms

RPCTwinMuxLinkMapPopConAnalyzer = cms.EDAnalyzer('RPCAMCLinkMapPopConAnalyzer'
                                                 , record = cms.string('RPCTwinMuxLinkMapRcd')
                                                 , Source = cms.PSet(
                                                     identifier = cms.string('RPCTwinMuxLinkMapHandler')
                                                     , dataTag = cms.string('RPCTwinMuxLinkMap_v1')
                                                     , sinceRun = cms.uint64(1)
                                                     # File provided by K. Bunkowski
                                                     , inputFile = cms.FileInPath('CondTools/RPC/data/RPCTwinMuxLinkMapInput.txt')
                                                     , wheelNotSide = cms.bool(True)
                                                     , wheelOrSideFED = cms.vint32(1395, 1391, 1390, 1393, 1394)
                                                     , nSectors = cms.uint32(12)
                                                     , wheelOrSideSectorAMC = cms.vint64(0x123456789ABC, 0x123456789ABC, 0x123456789ABC, 0x123456789ABC, 0x123456789ABC)
                                                     , txtFile = cms.untracked.string('')
                                                 )
)
