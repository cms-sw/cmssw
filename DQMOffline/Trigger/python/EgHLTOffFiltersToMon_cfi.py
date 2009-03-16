import FWCore.ParameterSet.Config as cms

egHLTOffFiltersToMon = cms.PSet (
    eleHLTFilterNames=cms.vstring("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter",
                                                               "hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter",
                                                               "hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter",
                                                               "hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter"),
                                 
    phoHLTFilterNames=cms.vstring("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter",
                                  "hlt1jet30"),
    eleTightLooseTrigNames=cms.vstring('hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter:hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter',
                                       'hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter',
                                       'hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter:hlt1jet30'),
    phoTightLooseTrigNames=cms.vstring('hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter:hlt1jet30'),
    diEleTightLooseTrigNames=cms.vstring('hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter:hltPreMinBiasEcal'),
    diPhoTightLooseTrigNames=cms.vstring(),
    
    )
