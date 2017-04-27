import FWCore.ParameterSet.Config as cms
ECALpedestalPCLHarvester = cms.EDAnalyzer('ECALpedestalPCLHarvester',
                                          MinEntries = cms.int32(100), #skip channel if stat is low
                                          ChannelStatusToExclude = cms.vstring('kDAC',
                                                                               'kNoisy',
                                                                               'kNNoisy',
                                                                               'kFixedG6',
                                                                               'kFixedG1',
                                                                               'kFixedG0',
                                                                               'kNonRespondingIsolated',
                                                                               'kDeadVFE',
                                                                               'kDeadFE',
                                                                               'kNoDataNoTP',)
                                          )
