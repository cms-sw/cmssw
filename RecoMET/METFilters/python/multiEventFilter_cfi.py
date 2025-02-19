
import FWCore.ParameterSet.Config as cms

multiEventFilter = cms.EDFilter(
  "MultiEventFilter",
  EventList = cms.vstring(
    "0:0:0"  # run:lumi:event
  ),
  taggingMode   = cms.bool(False),
  file = cms.FileInPath('RecoMET/METFilters/data/dummy.txt'),
)

vetoIncMuons = multiEventFilter.clone(EventList = [
 "142422:927:564419326",
 "143953:451:499461209",
 "147114:333:265212529",
 "149003:194:198684423",
 "149181:1790:1692590330",
 "149291:570:598587553",
 "149291:759:766167308"
])
