import FWCore.ParameterSet.Config as cms

RPCLinkMapSource = cms.ESSource('PoolDBESSource'
                                , DBParameters = cms.PSet()
                                , timetype = cms.string('timestamp')
                                , toGet = cms.VPSet(
                                    cms.PSet(
                                        record = cms.string('RPCLBLinkMapRcd')
                                        , tag = cms.string('RPCLBLinkMap_L1T_v1')
                                    )
                                    , cms.PSet(
                                        record = cms.string('RPCDCCLinkMapRcd')
                                        , tag = cms.string('RPCDCCLinkMap_L1T_v1')
                                    )
                                    , cms.PSet(
                                        record = cms.string('RPCTwinMuxLinkMapRcd')
                                        , tag = cms.string('RPCTwinMuxLinkMap_L1T_v1')
                                    )
                                    , cms.PSet(
                                        record = cms.string('RPCCPPFLinkMapRcd')
                                        , tag = cms.string('RPCCPPFLinkMap_L1T_v1')
                                    )
                                    , cms.PSet(
                                        record = cms.string('RPCOMTFLinkMapRcd')
                                        , tag = cms.string('RPCOMTFLinkMap_L1T_v1')
                                    )
                                )
                                , connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
)
