import FWCore.ParameterSet.Config as cms

RPCCPPFLinkMapPopConAnalyzer = cms.EDAnalyzer('RPCAMCLinkMapPopConAnalyzer'
                                              , record = cms.string('RPCCPPFLinkMapRcd')
                                              , Source = cms.PSet(
                                                  identifier = cms.string('RPCCPPFLinkMapHandler')
                                                  , dataTag = cms.string('RPCCPPFLinkMap_v1')
                                                  , sinceRun = cms.uint64(1)
                                                  # File provided by K. Bunkowski
                                                  , inputFile = cms.FileInPath('CondTools/RPC/data/RPCCPPFLinkMapInput.txt')
                                                  , wheelNotSide = cms.bool(False)
                                                  , wheelOrSideFED = cms.vint32(1386, 1386)
                                                  , nSectors = cms.uint32(4)
                                                  , wheelOrSideSectorAMC = cms.vint64(0x789a, 0x3456)
                                                  , txtFile = cms.untracked.string('')
                                              )
)
