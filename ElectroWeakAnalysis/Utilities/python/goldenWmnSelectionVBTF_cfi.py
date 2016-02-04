import FWCore.ParameterSet.Config as cms

# Trigger requirements
import HLTrigger.HLTfilters.hltHighLevel_cfi
wmnHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
wmnHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
wmnHLTFilter.HLTPaths = ["HLT_Mu9", "HLT_Mu11", "HLT_Mu15"]
wmnHLTFilter.throw = cms.bool(False)

# Cuts for each muon
goodMuonsForW = cms.EDFilter("MuonViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('isGlobalMuon=1 && isTrackerMuon=1 && abs(eta)<2.1 && abs(globalTrack().dxy)<0.2 && pt>20. && globalTrack().normalizedChi2<10 && globalTrack().hitPattern().numberOfValidTrackerHits>10 && globalTrack().hitPattern().numberOfValidMuonHits>0 && globalTrack().hitPattern().numberOfValidPixelHits>0 && numberOfMatches>1 && (isolationR03().sumPt+isolationR03().emEt+isolationR03().hadEt)<0.15*pt'),
  filter = cms.bool(True)
)

# Cuts on wmn system
wmnCands = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('sqrt((daughter(0).pt+daughter(1).pt)*(daughter(0).pt+daughter(1).pt)-pt*pt)>50'),
    decay = cms.string("goodMuonsForW pfMet")
)
wmnFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("wmnCands"),
    minNumber = cms.uint32(1)
)

# Dimuons to be vetoed
goodMuonsForZ = cms.EDFilter("MuonViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('isGlobalMuon=1 && pt>10.'),
  filter = cms.bool(True)
)
dyFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goodMuonsForZ"),
    minNumber = cms.uint32(2)
)

# Path
goldenWmnSequence  = cms.Sequence(
     wmnHLTFilter *
     goodMuonsForW *
     wmnCands*wmnFilter *
     goodMuonsForZ*~dyFilter
)
