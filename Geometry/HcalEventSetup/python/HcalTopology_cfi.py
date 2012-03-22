import FWCore.ParameterSet.Config as cms

HcalTopologyIdealEP = cms.ESProducer("HcalTopologyIdealEP",
                                     Exclude  = cms.untracked.string(''),
                                     H2Mode   = cms.untracked.bool(False),
                                     SLHCMode = cms.untracked.bool(False),
                                     H2HEMode = cms.untracked.bool(False),
)
