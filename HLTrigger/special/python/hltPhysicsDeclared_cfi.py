# coding: utf-8

import FWCore.ParameterSet.Config as cms

hltPhysicsDeclared = cms.EDFilter("HLTPhysicsDeclared",
  invert               = cms.bool( False ),
  L1GtReadoutRecordTag = cms.InputTag('hltGtDigis')
)
