import FWCore.ParameterSet.Config as cms

EcalTrivialConditionRetriever = cms.ESSource("EcalTrivialConditionRetriever",
    getLaserAlphaFromFile = cms.untracked.bool(True),
    EBLaserAlphaFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/barrel_ly.txt'),
    EELaserAlphaFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/endcap_ly.txt')
)
