import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorTrack.SiStripMonitorMuonHLT_cfi import *

hltHighLevelSiStrip = cms.EDFilter("HLTHighLevel",
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    HLTPaths = cms.vstring('HLT_L1Mu[^_]*$',
                           'HLT_L2Mu[^_]*$',
			   'HLT_Mu[^_]*$',
			   'HLT_IsoMu[^_]*$',
			   'HLT_DoubleMu[^_]*$',
    ),

    eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
    andOr = cms.bool(True), # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
    throw = cms.bool(True)    # throw exception on unknown path names
)

hltLocalRecoSiStrip = cms.Path(hltHighLevelSiStrip*sistripMonitorMuonHLT)

