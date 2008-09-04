import FWCore.ParameterSet.Config as cms

# $Id: L1Scalers_cfi.py,v 1.3 2008/09/03 10:35:27 lorenzo Exp $

l1s = cms.EDFilter("L1Scalers",
                   l1GtData = cms.InputTag("l1GtUnpack","","HLT"),
                   dqmFolder = cms.untracked.string("L1T/L1Scalers_EvF"),
                   verbose = cms.untracked.bool(False)
                   )
