import FWCore.ParameterSet.Config as cms

hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

from Geometry.HcalEventSetup.HcalRelabel_cfi import HcalReLabel
#------------------------- HARDCODED conditions

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
toGet = cms.untracked.vstring('GainWidths'),
#--- the following 5 parameters can be omitted in case of regular Geometry 
    iLumi = cms.double(-1.),                 # for Upgrade: fb-1
    HcalReLabel =  HcalReLabel,              # for Upgrade
    HERecalibration = cms.bool(False),       # True for Upgrade   
    HFRecalibration = cms.bool(False),       # True for Upgrade
    GainWidthsForTrigPrims = cms.bool(False) # True Upgrade    
)


es_prefer_hcalHardcode = cms.ESPrefer("HcalHardcodeCalibrations", "es_hardcode")

import Geometry.HcalEventSetup.hcalTopologyConstants_cfi as hcalTopologyConstants_cfi
es_hardcode.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)
