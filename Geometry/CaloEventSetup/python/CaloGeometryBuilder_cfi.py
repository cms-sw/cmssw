import FWCore.ParameterSet.Config as cms

CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL'          , 
                                'ZDC'           ,
                                'CASTOR'        ,
                                'EcalBarrel'    , 
                                'EcalEndcap'    , 
                                'EcalPreshower' , 
                                'TOWER'           )
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(CaloGeometryBuilder,
                     SelectedCalos = ['HCAL', 'ZDC', 'EcalBarrel', 'EcalEndcap', 'EcalPreshower', 'TOWER']
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(CaloGeometryBuilder,
                       SelectedCalos = ['HCAL', 'ZDC', 'EcalBarrel', 'TOWER', 'HGCalEESensitive', 'HGCalHESiliconSensitive', 'HGCalHEScintillatorSensitive']
)
