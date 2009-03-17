import FWCore.ParameterSet.Config as cms

MonitorHcalIsoTrackAlCaReco = cms.EDAnalyzer("DQMHcalIsoTrackAlCaReco",
folderName=cms.string("AlCaReco/HcalIsoTrack"),
saveToFile=cms.bool(True),
outputRootFileName=cms.string("HcalIsoTrackAlCaRecoMon.root"),
hltTriggerEventLabel=cms.InputTag('hltTriggerSummaryAOD'),
hltL3FilterLabel=cms.InputTag('hltIsolPixelTrackFilter::HLT'),
alcarecoIsoTracksLabel=cms.InputTag('IsoProd:HcalIsolatedTrackCollection'),
recoTracksLabel=cms.InputTag('IsoProd:IsoTrackTracksCollection')
)



