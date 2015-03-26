import FWCore.ParameterSet.Config as cms

EcalTrivialConditionRetriever = cms.ESSource("EcalTrivialConditionRetriever",
  producedEcalClusterLocalContCorrParameters = cms.untracked.bool(True),
  producedEcalClusterCrackCorrParameters = cms.untracked.bool(True),
  producedEcalClusterEnergyUncertaintyParameters = cms.untracked.bool(True),
  producedEcalClusterEnergyCorrectionParameters = cms.untracked.bool(True),
  producedEcalClusterEnergyCorrectionObjectSpecificParameters = cms.untracked.bool(True),

  producedEcalSamplesCorrelation = cms.untracked.bool(True),
  getSamplesCorrelationFromFile = cms.untracked.bool(True),
  SamplesCorrelationFile = cms.untracked.string('CondTools/Ecal/python/EcalSamplesCorrelation.txt')
)
