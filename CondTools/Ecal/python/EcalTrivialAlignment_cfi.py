import FWCore.ParameterSet.Config as cms

EcalTrivialConditionRetriever = cms.ESSource("EcalTrivialConditionRetriever",
    getEEAlignmentFromFile = cms.untracked.bool(True),
    EEAlignmentFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/myEEAlignment_2011.txt'),
    getESAlignmentFromFile = cms.untracked.bool(True),
    ESAlignmentFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/ESAlignment_2010.txt'),
    getEBAlignmentFromFile = cms.untracked.bool(True),
    EBAlignmentFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/myEBAlignment_2011.txt'),
    producedEcalClusterLocalContCorrParameters = cms.untracked.bool(True),
    producedEcalClusterCrackCorrParameters = cms.untracked.bool(True),
    producedEcalClusterEnergyCorrectionParameters = cms.untracked.bool(True),
    producedEcalClusterEnergyUncertaintyParameters = cms.untracked.bool(True),
    producedEcalClusterEnergyCorrectionObjectSpecificParameters = cms.untracked.bool(True)
)
