import FWCore.ParameterSet.Config as cms

egHLTOffDQMClient = cms.EDFilter("EgHLTOfflineClient",
                                 sourceModuleName = cms.string('egammaHLTDQM'),
                                 DQMDirName=cms.string("HLT/EgammaHLTOffline_egammaHLTDQM"),
                                 eleHLTPathNames=cms.vstring("hltL1NonIsoHLTNonIsoSingleElectronEt15",
                                                             "hltL1NonIsoHLTNonIsoSingleElectronLWEt15"),
                                 eleHLTFilterNames=cms.vstring("TrackIsolFilter"),
                                 eleHLTTightLooseFilters=cms.vstring("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter" )
                                 )


