import FWCore.ParameterSet.Config as cms

RPCOMTFLinkMapPopConAnalyzer = cms.EDAnalyzer("RPCAMCLinkMapPopConAnalyzer"
                                              , record = cms.string("RPCOMTFLinkMapRcd")
                                              , Source = cms.PSet(
                                                  identifier = cms.string("RPCOMTFLinkMapHandler")
                                                  , dataTag = cms.string("RPCOMTFLinkMap_v1")
                                                  , sinceRun = cms.uint64(1)
                                                  # File provided by K. Bunkowski
                                                  , inputFile = cms.FileInPath("CondTools/RPC/data/RPCOMTFLinkMapInput.txt")
                                                  , wheelNotSide = cms.bool(False)
                                                  , wheelOrSideFED = cms.vint32(1380, 1381)
                                                  , nSectors = cms.uint32(6)
                                                  , wheelOrSideSectorAMC = cms.vint64(0x13579B, 0x13579B)
                                                  , txtFile = cms.untracked.string("")
                                              )
)
