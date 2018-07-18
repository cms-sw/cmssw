import FWCore.ParameterSet.Config as cms

siStripLorentzAngleFakeESSource = cms.ESSource("SiStripLorentzAngleFakeESSource",
       appendToDataLabel = cms.string(''),

       # Three possible generations:
       # - give two values = (min,max)                                                -> uniform distribution
       # - give one value and PerCent_Err != 0                                        -> gaussian distribution
       # - either give two equal values or a single value (pass an empty max vector)  -> fixed value
       
       # TIB min and max
       TIB_EstimatedValuesMin = cms.vdouble(0.014, 0.014, 0.014, 0.014),
       TIB_EstimatedValuesMax = cms.vdouble(),
       # TIB errors
       TIB_PerCent_Errs       = cms.vdouble(0.,    0.,    0.,    0.),
       # TOB min and max
       TOB_EstimatedValuesMin = cms.vdouble(0.021, 0.021, 0.021, 0.021, 0.021, 0.021),
       TOB_EstimatedValuesMax = cms.vdouble(0.021, 0.021, 0.021, 0.021, 0.021, 0.021),
       # TOB errors
       TOB_PerCent_Errs       = cms.vdouble(0.,    0.,    0.,    0.,    0.,    0.),
   )
