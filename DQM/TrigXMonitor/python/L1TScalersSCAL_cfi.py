import FWCore.ParameterSet.Config as cms

#demo = cms.EDAnalyzer('L1TScalers'
#)

l1tscalers = cms.EDAnalyzer("L1TScalersSCAL",
                   #l1GtData = cms.InputTag("l1GtUnpack","","HLT"),
                   #dqmFolder = cms.untracked.string("L1T/L1Scalers_EvF"),
                   verbose = cms.untracked.bool(False),
                   #firstFED = cms.untracked.uint32(0),
                   #lastFED = cms.untracked.uint32(39),
		   denomIsTech = cms.untracked.bool(True),
                   denomBit = cms.untracked.uint32(35),
                   muonBit = cms.untracked.uint32(55),
                   egammaBit = cms.untracked.uint32(46),
                   jetBit = cms.untracked.uint32(16),
                   #fedRawData = cms.InputTag("source","", ""),
                   #maskedChannels = cms.untracked.vint32(),
                   #HFRecHitCollection = cms.InputTag("hfreco", "", "")
                   #scalersResults = cms.InputTag("l1scalers","","HLT")                                     	
                   scalersResults = cms.InputTag("scalersRawToDigi","","DQM")                                     	
)
                                                                                         
