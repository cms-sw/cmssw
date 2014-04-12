import FWCore.ParameterSet.Config as cms

import copy

# Uncomment to use trigger requirements
#import HLTrigger.HLTfilters.hltHighLevel_cfi
#zmmHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
#zmmHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
#zmmHLTFilter.HLTPaths = ["HLT_Mu9", "HLT_Mu11", "HLT_Mu15", "HLT_DoubleMu3"]
#zmmHLTFilter.throw = cms.bool(False)

# Cuts for each muon
goodMuons1 = cms.EDFilter("MuonViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('isGlobalMuon=1 && isTrackerMuon=1 && abs(eta)<2.1 && abs(globalTrack().dxy)<0.2 && pt>20. && globalTrack().normalizedChi2<10 && globalTrack().hitPattern().numberOfValidTrackerHits>10 && globalTrack().hitPattern().numberOfValidMuonHits>0 && globalTrack().hitPattern().numberOfValidPixelHits>0 && numberOfMatches>1 && (isolationR03().sumPt+isolationR03().emEt+isolationR03().hadEt)<0.15*pt'),
  filter = cms.bool(True)
)

# Cuts for each muon
goodMuons2 = cms.EDFilter("MuonViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('isGlobalMuon=1 && pt>20. && abs(eta)<2.4 && abs(globalTrack().dxy)<1.0 && globalTrack().hitPattern().numberOfValidTrackerHits>6'),
  filter = cms.bool(True)
)

# Cuts on dimuon system
zmmCands = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass>60'),
    decay = cms.string("goodMuons1 goodMuons2")
)
zmmFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("zmmCands"),
    minNumber = cms.uint32(1)
)

# Selection sequence
goldenZmmSelectionSequence = cms.Sequence(
     #zmmHLTFilter *
     goodMuons1*goodMuons2 *
     zmmCands*zmmFilter
)
