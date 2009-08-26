import FWCore.ParameterSet.Config as cms

#currently not complete menu

egHLTOffFiltersToMon = cms.PSet (
    eleHLTFilterNames=cms.vstring(
                                  "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdDphiFilter",  #EleEt15SWEleId                                
                                  "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdTrackIsolFilterESet25LTI", #EleEt15SWEleId with track isol 0.5 + 8*pt
                                  "hltL1NonIsoHLTNonIsoSingleElectronEt20PixelMatchFilter", #Ele 20
                                
                                  ),
                                  
    phoHLTFilterNames=cms.vstring(),
    eleTightLooseTrigNames=cms.vstring("hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdTrackIsolFilterESet25LTI:hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter"),
                                      
    phoTightLooseTrigNames=cms.vstring(),
    diEleTightLooseTrigNames=cms.vstring(),
    diPhoTightLooseTrigNames=cms.vstring(),
    
    )
