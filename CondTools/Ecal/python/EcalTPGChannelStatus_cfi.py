import FWCore.ParameterSet.Config as cms

EcalTrivialConditionRetriever = cms.ESSource("EcalTrivialConditionRetriever",
  producedEcalTrgChannelStatus = cms.untracked.bool(True),
#  trgChannelStatusFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/trgChannelStatus.txt')
#  trgChannelStatusFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/triggermasks_confid722.txt')
  trgChannelStatusFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/triggermasks_confid722_plus_ebm10_tt64.txt')
)
