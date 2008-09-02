import FWCore.ParameterSet.Config as cms

# $Id$

l1s = cms.EDFilter("L1Scalers",
                   l1GtData = cms.InputTag("l1GtUnpack","","HLT"),
                   scalersResults = cms.InputTag("l1scalers","","HLT"),
                   verbose = cms.untracked.bool(False)
                   )
