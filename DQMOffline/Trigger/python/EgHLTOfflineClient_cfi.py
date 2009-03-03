import FWCore.ParameterSet.Config as cms

egHLTOffDQMClient = cms.EDFilter("EgHLTOfflineClient",
                                 sourceModuleName = cms.string('egammaHLTDQM'),
                                 DQMDirName=cms.string("HLT/EgammaHLTOffline_egammaHLTDQM"),
                                 eleHLTFilterNames=cms.vstring("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter",
                                                              "hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter",
                                                               "hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter",
                                                               "hlt1jet30"),
                                 phoHLTFilterNames=cms.vstring("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter",
                                                               "hlt1jet30"),
                                 eleTightLooseTrigNames=cms.vstring('hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter:hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter',
                                                                    'hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter',
                                                                    'hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter:hlt1jet30',
                                                                    'hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter:hlt1jet50'),
                                 phoTightLooseTrigNames=cms.vstring('hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter:hlt1jet30'),
                                 eleEffVars=cms.vstring('dEtaIn','dPhiIn','hOverE','sigmaEtaEta'),
                                 phoEffVars=cms.vstring('hOverE','sigmaEtaEta'),
                                 eleTrigTPEffVsVars=cms.vstring('et','eta','phi','charge'),
                                 phoTrigTPEffVsVars=cms.vstring('et','eta','phi','charge'),
                                 eleLooseTightTrigEffVsVars=cms.vstring('et','eta','phi','charge',
                                                                        'hOverE','sigmaEtaEta','dPhiIn','dEtaIn'),
                                 phoLooseTightTrigEffVsVars=cms.vstring('et','eta','phi','charge',
                                                                        'hOverE','sigmaEtaEta')
                                 )


