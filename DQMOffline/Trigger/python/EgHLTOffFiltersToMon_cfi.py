import FWCore.ParameterSet.Config as cms

#currently not complete menu

egHLTOffFiltersToMon = cms.PSet (

    eleHLTFilterNames=cms.vstring(

    "hltPreL1SingleEG5",
    "hltPreL1SingleEG8",
    #8E29
    #ele
    "hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter",
   
    #double ele
    "hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter",


    #photon
    "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoSinglePhotonEt15HTITrackIsolFilter",
    "hltL1NonIsoSinglePhotonEt15LEIHcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter",
    #double pho
    "hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter",    
    "hltL1NonIsoDoublePhotonEt5eeResPMMassFilter",
    "hltL1NonIsoDoublePhotonEt5JpsiPMMassFilter",
    "hltL1NonIsoDoublePhotonEt5UpsPMMassFilter",


    

    #1E31
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
    "hltPreL1SingleEG5",
    "hltPreL1SingleEG8",
    #8E29
    "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoSinglePhotonEt15HTITrackIsolFilter",
    "hltL1NonIsoSinglePhotonEt15LEIHcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter",    
    "hltL1NonIsoDoublePhotonEt5eeResPMMassFilter",
    "hltL1NonIsoDoublePhotonEt5JpsiPMMassFilter",
    "hltL1NonIsoDoublePhotonEt5UpsPMMassFilter",

    #1E31
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

    #8E29
    #ele triggers
    "hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter:hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    #"hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter:hltPreL1DoubleEG5",

    #photon
    "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter:hltPreL1SingleEG5",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter:hltPreL1SingleEG8",
    "hltL1NonIsoSinglePhotonEt15HTITrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoSinglePhotonEt15LEIHcalIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter:hltPreL1SingleEG8",
    


    #1E31
    #ele triggers
    "hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdDphiFilter:hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15LTITrackIsolFilter:hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdTrackIsolFilterESet25LTI:hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdDphiFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt20PixelMatchFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    
    

    #pho triggers
    "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter:hltPreL1SingleEG5",
    #"hltL1NonIsoHLTLEITISinglePhotonEt10TrackIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter:hltPreL1SingleEG5",
    "hltL1NonIsoHLTLEITISinglePhotonEt20TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter:hltPreL1SingleEG5",
    "hltL1NonIsoHLTLEITISinglePhotonEt25TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter:hltPreL1SingleEG5",
    ),
                                      
    phoTightLooseTrigNames=cms.vstring(

    #photon 8E29
    "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter:hltPreL1SingleEG5",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter:hltPreL1SingleEG8",
    "hltL1NonIsoSinglePhotonEt15HTITrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoSinglePhotonEt15LEIHcalIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter:hltPreL1SingleEG8",
    
    #pho triggers 1E31
    "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter:hltPreL1SingleEG5",
    #"hltL1NonIsoHLTLEITISinglePhotonEt10TrackIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter:hltPreL1SingleEG5",
    "hltL1NonIsoHLTLEITISinglePhotonEt20TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter:hltPreL1SingleEG5",
    "hltL1NonIsoHLTLEITISinglePhotonEt25TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter:hltPreL1SingleEG5",
    ),

    
    diEleTightLooseTrigNames=cms.vstring(
    #8E29 ele
     "hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter:hltL1NonIsoDoublePhotonEt5eeResPMMassFilter",
     "hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter:hltL1NonIsoDoublePhotonEt5JpsiPMMassFilter",
     "hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter:hltL1NonIsoDoublePhotonEt5UpsPMMassFilter",
     #pho
     "hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter:hltPreL1DoubleEG5"
     "hltL1NonIsoDoublePhotonEt5eeResPMMassFilter:hltPreL1DoubleEG5",
     "hltL1NonIsoDoublePhotonEt5JpsiPMMassFilter:hltPreL1DoubleEG5",
     "hltL1NonIsoDoublePhotonEt5UpsPMMassFilter:hltPreL1DoubleEG5",

     #1E31 ele
     "hltL1NonIsoHLTNonIsoDoubleElectronEt10PixelMatchFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter",
     #1E31 pho
     "hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter",
     "hltL1NonIsoHLTVLEIDoublePhotonEt15HcalIsolFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter",
     "hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter",
   
     ),
    diPhoTightLooseTrigNames=cms.vstring(
    #8E29 pho
    "hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter:hltPreL1DoubleEG5"
    "hltL1NonIsoDoublePhotonEt5eeResPMMassFilter:hltPreL1DoubleEG5",
    "hltL1NonIsoDoublePhotonEt5JpsiPMMassFilter:hltPreL1DoubleEG5",
    "hltL1NonIsoDoublePhotonEt5UpsPMMassFilter:hltPreL1DoubleEG5",
      #1E31 pho
    "hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTVLEIDoublePhotonEt15HcalIsolFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter:hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter",
   
    ),
    
    )
