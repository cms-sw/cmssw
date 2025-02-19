import FWCore.ParameterSet.Config as cms

import copy

# Trigger requirements
import HLTrigger.HLTfilters.hltHighLevel_cfi
zmmHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zmmHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
zmmHLTFilter.HLTPaths = ["HLT_Mu9", "HLT_Mu11", "HLT_Mu15"]
zmmHLTFilter.throw = cms.bool(False)

# Cuts for both muons
goodMuons = cms.EDFilter("MuonViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('pt>20. && abs(eta)<2.1 && isGlobalMuon=1 && isTrackerMuon=1 && abs(globalTrack().dxy)<0.2 && globalTrack().normalizedChi2<10 && globalTrack().hitPattern().numberOfValidTrackerHits>10 && globalTrack().hitPattern().numberOfValidMuonHits>0 && globalTrack().hitPattern().numberOfValidPixelHits>0 && numberOfMatches>1 && (isolationR03().sumPt+isolationR03().emEt+isolationR03().hadEt)<0.15*pt'),
  filter = cms.bool(True)
)

# Cuts on dimuon system
zmmCands = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string('mass>60 && mass<120 && charge=0'),
    decay = cms.string("goodMuons@+ goodMuons@-")
)
zmmFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("zmmCands"),
    minNumber = cms.uint32(1)
)

# Selection sequence
goldenZmmSelectionSequence = cms.Sequence(
     zmmHLTFilter *
     goodMuons *
     zmmCands*zmmFilter
)
