import FWCore.ParameterSet.Config as cms

CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL'          , 
                                'ZDC'           ,
                                'CASTOR'        ,
                                'EcalBarrel'    , 
                                'EcalEndcap'    , 
                                'EcalShashlik'    , 
                                'EcalPreshower' , 
                                'TOWER'           )
)


