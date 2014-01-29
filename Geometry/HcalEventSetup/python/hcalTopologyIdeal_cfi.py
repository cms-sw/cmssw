import FWCore.ParameterSet.Config as cms

hcalTopologyIdeal = cms.ESProducer('HcalTopologyIdealEP',
  Exclude = cms.untracked.string(''),
##  hcalTopologyConstants = cms.PSet(),
  appendToDataLabel = cms.string('')
)
