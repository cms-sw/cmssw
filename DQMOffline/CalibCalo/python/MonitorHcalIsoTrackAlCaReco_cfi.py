import FWCore.ParameterSet.Config as cms

MonitorHcalIsoTrackAlCaReco = cms.EDAnalyzer("DQMHcalIsoTrackAlCaReco",
folderName=cms.string("AlCaReco/HcalIsoTrack"),
outputRootFileName=cms.string("HcalIsoTrackAlCaRecoMon.root"),
hltTriggerEventLabel=cms.InputTag('hltTriggerSummaryAOD'),
hltL3FilterLabel=cms.InputTag('hltIsolPixelTrackFilter::HLT'),
recoTracksLabel=cms.InputTag('IsoProd:IsoTrackTracksCollection')
)



