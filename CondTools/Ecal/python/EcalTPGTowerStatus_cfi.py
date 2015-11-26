import FWCore.ParameterSet.Config as cms

EcalTrivialConditionRetriever = cms.ESSource("EcalTrivialConditionRetriever",
  producedEcalTrgTowerStatus = cms.untracked.bool(True),
  trgTowerStatusFile = cms.untracked.string('CondTools/Ecal/python/BTT_DATA_CONF_ID_2982.dat')
)
