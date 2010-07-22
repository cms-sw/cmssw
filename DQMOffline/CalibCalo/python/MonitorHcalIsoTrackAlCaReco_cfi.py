import FWCore.ParameterSet.Config as cms

MonitorHcalIsoTrackAlCaReco = cms.EDAnalyzer("DQMHcalIsoTrackAlCaReco",
folderName=cms.string("AlCaReco/HcalIsoTrack"),
saveToFile=cms.bool(False),
outputRootFileName=cms.string("HcalIsoTrackAlCaRecoMon.root"),
hltTriggerEventLabel=cms.InputTag('hltTriggerSummaryAOD'),
hltL3FilterLabels=cms.vstring('hltIsolPixelTrackL3FilterHB','hltIsolPixelTrackL3FilterHE'),
filterNameLength=cms.int32(27),
alcarecoIsoTracksLabel=cms.InputTag('IsoProd:HcalIsolatedTrackCollection'),
recoTracksLabel=cms.InputTag('IsoProd:IsoTrackTracksCollection')
)



