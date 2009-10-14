import FWCore.ParameterSet.Config as cms

#currently not complete menu

egHLTOffFiltersToMon = cms.PSet (

    eleHLTFilterNames=cms.vstring(
    "hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdDphiFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15LTITrackIsolFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt20PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdTrackIsolFilterESet25LTI", #EleEt15SWEleId with track isol 0.5 + 8*pt
    #double ele
    "hltL1NonIsoDoubleElectronEt5JpsiPMMassFilter",
    "hltL1NonIsoDoubleElectronEt5UpsPMMassFilter",
    "hltL1NonIsoHLTNonIsoDoubleElectronEt10PixelMatchFilter",
   
    #photon triggers
    "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTLEITISinglePhotonEt10TrackIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTLEITISinglePhotonEt20TrackIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter",
    "hltL1NonIsoHLTLEITISinglePhotonEt25TrackIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter",
    #double pho
    "hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTVLEIDoublePhotonEt15HcalIsolFilter",
    ),
                                
    phoHLTFilterNames=cms.vstring(
    "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTLEITISinglePhotonEt10TrackIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTLEITISinglePhotonEt20TrackIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter",
    "hltL1NonIsoHLTLEITISinglePhotonEt25TrackIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter",
    #double pho
    "hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTVLEIDoublePhotonEt15HcalIsolFilter",
    
    ),

    eleTightLooseTrigNames=cms.vstring(
    #ele triggers
    "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdTrackIsolFilterESet25LTI:hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdDphiFilter:hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15LTITrackIsolFilter:hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdTrackIsolFilterESet25LTI:hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdDphiFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt20PixelMatchFilter:hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter",
    #pho triggers
    "hltL1NonIsoHLTLEITISinglePhotonEt25TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTLEITISinglePhotonEt20TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    
    ),
                                      
    phoTightLooseTrigNames=cms.vstring(
    #pho triggers
    "hltL1NonIsoHLTLEITISinglePhotonEt25TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTLEITISinglePhotonEt20TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    ),
    diEleTightLooseTrigNames=cms.vstring(
    "hltL1NonIsoHLTNonIsoDoubleElectronEt10PixelMatchFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter"
    "hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTVLEIDoublePhotonEt15HcalIsolFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter",
    ),
    diPhoTightLooseTrigNames=cms.vstring(
    
    "hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTVLEIDoublePhotonEt15HcalIsolFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter",
    ),
    
    )
