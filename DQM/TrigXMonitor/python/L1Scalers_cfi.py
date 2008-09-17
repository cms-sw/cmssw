import FWCore.ParameterSet.Config as cms

# $Id: L1Scalers_cfi.py,v 1.5 2008/09/16 17:19:13 wittich Exp $

l1s = cms.EDFilter("L1Scalers",
                   l1GtData = cms.InputTag("l1GtUnpack","","HLT"),
                   dqmFolder = cms.untracked.string("L1T/L1Scalers_EvF"),
                   verbose = cms.untracked.bool(False),
                   firstFED = cms.untracked.uint32(0),
                   lastFED = cms.untracked.uint32(39),
                   fedRawData = cms.InputTag("source","", ""),
                   maskedChannels = cms.untracked.vint32(),
                   HFRecHitCollection = cms.InputTag("hfreco", "", "")
                   )
