import FWCore.ParameterSet.Config as cms

RPCLinkMapSource = cms.ESSource('PoolDBESSource'
                                , DBParameters = cms.PSet()
                                , timetype = cms.string('timestamp')
                                , toGet = cms.VPSet(
                                    cms.PSet(
                                        record = cms.string('RPCLBLinkMapRcd')
                                        , tag = cms.string('RPCLBLinkMap_v1')
                                    )
                                    , cms.PSet(
                                        record = cms.string('RPCDCCLinkMapRcd')
                                        , tag = cms.string('RPCDCCLinkMap_v1')
                                    )
                                    , cms.PSet(
                                        record = cms.string('RPCTwinMuxLinkMapRcd')
                                        , tag = cms.string('RPCTwinMuxLinkMap_v1')
                                    )
                                    , cms.PSet(
                                        record = cms.string('RPCCPPFLinkMapRcd')
                                        , tag = cms.string('RPCCPPFLinkMap_v1')
                                    )
                                    , cms.PSet(
                                        record = cms.string('RPCOMTFLinkMapRcd')
                                        , tag = cms.string('RPCOMTFLinkMap_v1')
                                    )
                                )
                                , connect = cms.string('sqlite_fip:CondTools/RPC/data/RPCLinkMap.db')
)
