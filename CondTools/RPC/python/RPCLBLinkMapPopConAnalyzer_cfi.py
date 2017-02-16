import FWCore.ParameterSet.Config as cms

RPCLBLinkMapPopConAnalyzer = cms.EDAnalyzer('RPCLBLinkMapPopConAnalyzer'
                                            , record = cms.string('RPCLBLinkMapRcd')
                                            , Source = cms.PSet(
                                                identifier = cms.string('RPCLBLinkMapHandler')
                                                , dataTag = cms.string('RPCLBLinkMap_v1')
                                                , sinceRun = cms.uint64(1)
                                                , DBParameters = cms.PSet(
                                                    authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
                                                    , authenticationSystem = cms.untracked.int32(2)
                                                    , security = cms.untracked.string('')
                                                    , messageLevel = cms.untracked.int32(0)
                                                )
                                                , connect = cms.string('oracle://cms_omds_adg/CMS_RPC_COND')
                                                , txtFile = cms.untracked.string('')
                                            )
)
