# The following comments couldn't be translated into the new config version:

#
#
# Modules for soft Jet trigger skim. Designed for CSA07 only.
# Should be used in veto mode and kill dominant samples of other physics
#

import FWCore.ParameterSet.Config as cms

singleJetTrigger = cms.EDFilter("QCDSingleJetFilter",
    TriggerJetCollectionB = cms.InputTag("midPointCone7CaloJets"),
    MinPt = cms.double(20.0),
    TriggerJetCollectionA = cms.InputTag("midPointCone7CaloJets")
)

muonTrigger = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_Mu15_L1Mu7'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

electronTrigger = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_IsoEle18_L1R'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

photonTrigger = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_IsoPhoton40_L1R'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


