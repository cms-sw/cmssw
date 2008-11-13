import FWCore.ParameterSet.Config as cms

# File: TCMET.cff
# Author: R.Remington
# Date: 11.12.2008
#
# Form Track Corrected MET

tcMet = cms.EDProducer("METProducer",
    src = cms.InputTag("towerMaker"),
    METType = cms.string('MET'),
    alias = cms.string('TCMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection')
)



