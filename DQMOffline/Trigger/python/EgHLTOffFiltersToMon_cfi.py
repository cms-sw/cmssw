import FWCore.ParameterSet.Config as cms

#currently not complete menu

egHLTOffFiltersToMon = cms.PSet (

    eleHLTFilterNames=cms.vstring(

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

    #si strip
    "hltL1NonIsoHLTNonIsoSingleElectronSiStripEt15PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronSiStripEt20PixelMatchFilter",
    
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

    #----Morse----
    #5E32 and beyond

    #electron triggers
    #"hltEle8PixelMatchFilter", #HLT_Ele8_v1
    #"hltEle8CaloIdLCaloIsoVLPixelMatchFilter", #HLT_Ele8_CaloIdL_CaloIsoVL_v1
    #"hltEle8CaloIdLTrkIdVLDphiFilter", #HLT_Ele8_CaloIdL_TrkIdVL_v1
    "hltEle15CaloIdVTTrkIdTCaloIsoTTrkIsoTTrackIsolFilter", #HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v1
    "hltEle17CaloIdLCaloIsoVLPixelMatchFilter", #HLT_Ele17_CaloIdL_CaloIsoVL_v1
    "hltEle27CaloIdTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter", #HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v1
    "hltEle45CaloIdVTTrkIdTDphiFilter", #HLT_Ele45_CaloIdVT_TrkIdT_v1
    "hltEle90NoSpikeFilterPixelMatchFilter", #HLT_Ele90_NoSpikeFilter_v1     
    #"hltEle32CaloIdVLCaloIsoVLTrkIdVLTrkIsoVLTrackIsoFilter",
    "hltEle32CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter",
    "hltEle52CaloIdVTTrkIdTDphiFilter",
    "hltEle65CaloIdVTTrkIdTDphiFilter",
    "hltEle25CaloIdLCaloIsoVLTrkIdVLTrkIsoVLTrackIsoFilter",
    "hltEle25WP80PFMT40PFMTFilter",
    #"hltEle42CaloIdVLCaloIsoVLTrkIdVLTrkIsoVLTrackIsoFilter",
    "hltEle42CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter",

    #photon triggers
    "hltEG30CaloIdVLHEFilter", #HLT_Photon30_CaloIdVL_v1
    "hltPhoton30CaloIdVLIsoLTrackIsoFilter", #HLT_Photon30_CaloIdVL_IsoL_v1
    "hltPhoton50CaloIdVLIsoLTrackIsoFilter ",#HLT_Photon50_CaloIdVL_IsoL_v1
    "hltPhoton75CaloIdVLHEFilter", #HLT_Photon75_CaloIdVL_v1
    "hltPhoton75CaloIdVLIsoLTrackIsoFilter", #HLT_Photon75_CaloIdVL_IsoL_v1 
    "hltPhoton125HEFilter", #HLT_Photon125_NoSpikeFilter_v1   
    #double pho 
    "hltDoublePhoton33EgammaLHEDoubleFilter"#HLT_DoublePhoton33_v1

    #l1
 #   "hltPreL1SingleEG5",
 #   "hltPreL1SingleEG8",
 #   "hltPreL1SingleEG12",
  #  "hltPreL1SingleEG15",
 #   "hltPreL1SingleEG20",
    #----------------

    
    ),
                                
    phoHLTFilterNames=cms.vstring(

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

    #---Morse-------
    #5E32
    "hltEG30CaloIdVLHEFilter", #HLT_Photon30_CaloIdVL_v1
    "hltPhoton30CaloIdVLIsoLTrackIsoFilter", #HLT_Photon30_CaloIdVL_IsoL_v1
    "hltPhoton50CaloIdVLIsoLTrackIsoFilter ",#HLT_Photon50_CaloIdVL_IsoL_v1
    "hltPhoton75CaloIdVLHEFilter", #HLT_Photon75_CaloIdVL_v1
    "hltPhoton75CaloIdVLIsoLTrackIsoFilter", #HLT_Photon75_CaloIdVL_IsoL_v1 
    "hltPhoton125HEFilter", #HLT_Photon125_NoSpikeFilter_v1
    "hltPhoton50CaloIdVLHEFilter",
    "hltPhoton90CaloIdVLHEFilter",
    "hltPhoton90CaloIdVLIsoLTrackIsoFilter",
    #double pho 
    "hltDoublePhoton33EgammaLHEDoubleFilter"#HLT_DoublePhoton33_v1
    
    #l1
 #   "hltPreL1SingleEG5",
 #   "hltPreL1SingleEG8",
 #   "hltPreL1SingleEG12",
 #   "hltPreL1SingleEG15",
 #   "hltPreL1SingleEG20",
    #------------
   
    ),

    eleTightLooseTrigNames=cms.vstring(

    #8E29
    #ele triggers
    "hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter:hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter:hltL1NonIsoHLTNonIsoSingleElectronSiStripEt15PixelMatchFilter",
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
    "hltL1NonIsoHLTNonIsoSingleElectronEt20PixelMatchFilter:hltL1NonIsoHLTNonIsoSingleElectronSiStripEt20PixelMatchFilter",
    "hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter:hltL1NonIsoHLTNonIsoSingleElectronSiStripEt15PixelMatchFilter",

    #pho triggers
    "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter:hltPreL1SingleEG5",
    #"hltL1NonIsoHLTLEITISinglePhotonEt10TrackIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter:hltPreL1SingleEG5",
    "hltL1NonIsoHLTLEITISinglePhotonEt20TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter:hltPreL1SingleEG5",
    "hltL1NonIsoHLTLEITISinglePhotonEt25TrackIsolFilter:hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter",
    "hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter:hltPreL1SingleEG5",

    #---Morse----------------
    #5E32
    #ele triggers
  #  "hltEle8PixelMatchFilter:hltPreL1SingleEG5",
  #  "hltEle8CaloIdLCaloIsoVLPixelMatchFilter:hltEle8PixelMatchFilter",
  #  "hltEle8CaloIdLTrkIdVLDphiFilter:hltEle8PixelMatchFilter",
  #  "hltEle17CaloIdLCaloIsoVLPixelMatchFilter:hltEle8CaloIdLCaloIsoVLPixelMatchFilter",    
  #  "hltEle27CaloIdTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter:hltEle15CaloIdVTTrkIdTCaloIsoTTrkIsoTTrackIsolFilter",
  #  "hltEle45CaloIdVTTrkIdTDphiFilter:hltEle8CaloIdLTrkIdVLDphiFilter",
    #pho triggers
  #  "hltPhoton30CaloIdVLIsoLTrackIsoFilter:hltEG30CaloIdVLHEFilter",
  #  "hltPhoton50CaloIdVLIsoLTrackIsoFilter:hltPhoton30CaloIdVLIsoLTrackIsoFilter",
  #  "hltPhoton75CaloIdVLHEFilter:hltEG30CaloIdVLHEFilter",
  #  "hltPhoton75CaloIdVLIsoLTrackIsoFilter:hltPhoton30CaloIdVLIsoLTrackIsoFilter",
  #  "hltPhoton75CaloIdVLIsoLTrackIsoFilter:hltPhoton75CaloIdVLHEFilter",
    #-------------------------


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

    
    #---Morse----------------
    #5E32
    #pho triggers
 #   "hltPhoton30CaloIdVLIsoLTrackIsoFilter:hltEG30CaloIdVLHEFilter",
 #   "hltPhoton50CaloIdVLIsoLTrackIsoFilter:hltPhoton30CaloIdVLIsoLTrackIsoFilter",
 #   "hltPhoton75CaloIdVLHEFilter:hltEG30CaloIdVLHEFilter",
 #   "hltPhoton75CaloIdVLIsoLTrackIsoFilter:hltPhoton30CaloIdVLIsoLTrackIsoFilter",
 #   "hltPhoton75CaloIdVLIsoLTrackIsoFilter:hltPhoton75CaloIdVLHEFilter",
    #-------------------------

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
