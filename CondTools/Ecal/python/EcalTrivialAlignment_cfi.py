import FWCore.ParameterSet.Config as cms

EcalTrivialConditionRetriever = cms.ESSource("EcalTrivialConditionRetriever",
  producedEcalClusterLocalContCorrParameters = cms.untracked.bool(True),
  producedEcalClusterCrackCorrParameters = cms.untracked.bool(True),
  producedEcalClusterEnergyUncertaintyParameters = cms.untracked.bool(True),
  producedEcalClusterEnergyCorrectionParameters = cms.untracked.bool(True),
  producedEcalClusterEnergyCorrectionObjectSpecificParameters = cms.untracked.bool(True),

  getEEAlignmentFromFile = cms.untracked.bool(True),
  EEAlignmentFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/EEAlignment_2015.txt'),
  getESAlignmentFromFile = cms.untracked.bool(True),
  ESAlignmentFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/ESAlignment_2015.txt'),
  getEBAlignmentFromFile = cms.untracked.bool(True),
  EBAlignmentFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/EBAlignment_2015.txt')
)
