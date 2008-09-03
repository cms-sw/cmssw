import FWCore.ParameterSet.Config as cms

# $Id: L1Scalers_cfi.py,v 1.1 2008/09/02 02:35:32 wittich Exp $

l1s = cms.EDFilter("L1Scalers",
                   l1GtData = cms.InputTag("l1GtUnpack","","HLT"),
                   scalersResults = cms.InputTag("l1scalers","","HLT"),
                   dqmFolder = cms.untracked.string("L1Scalers_EvF"),
                   verbose = cms.untracked.bool(False)
                   )
