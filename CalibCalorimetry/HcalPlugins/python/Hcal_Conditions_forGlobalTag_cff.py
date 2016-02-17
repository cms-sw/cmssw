import FWCore.ParameterSet.Config as cms

hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

#------------------------- HARDCODED conditions

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
toGet = cms.untracked.vstring('GainWidths'),
#--- the following 5 parameters can be omitted in case of regular Geometry 
    iLumi = cms.double(-1.),                 # for Upgrade: fb-1
    HERecalibration = cms.bool(False),       # True for Upgrade   
    HEreCalibCutoff = cms.double(20.),       # if above is True  
    HFRecalibration = cms.bool(False),       # True for Upgrade
    GainWidthsForTrigPrims = cms.bool(False),# True Upgrade   
    testHFQIE10 = cms.bool(False)            # True 2016
)

from Configuration.StandardSequences.Eras import eras
eras.run2_HF_2016.toModify( es_hardcode, testHFQIE10=cms.bool(True) )

es_prefer_hcalHardcode = cms.ESPrefer("HcalHardcodeCalibrations", "es_hardcode")

