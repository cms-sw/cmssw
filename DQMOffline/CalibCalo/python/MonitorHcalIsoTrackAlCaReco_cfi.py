import FWCore.ParameterSet.Config as cms

MonitorHcalIsoTrackAlCaReco = cms.EDAnalyzer("DQMHcalIsoTrackAlCaReco",
folderName=cms.string("AlCaReco/HcalIsoTrack"),
saveToFile=cms.bool(False),
outputRootFileName=cms.string("HcalIsoTrackAlCaRecoMon.root"),
hltTriggerEventLabel=cms.InputTag('hltTriggerSummaryAOD'),
l1FilterLabel=cms.string('hltL1sJet52'),
hltL3FilterLabels=cms.vstring('hltIsolPixelTrackL3FilterHB','hltIsolPixelTrackL3FilterHE'),
alcarecoIsoTracksLabel=cms.InputTag('IsoProd:HcalIsolatedTrackCollection'),
recoTracksLabel=cms.InputTag('IsoProd:IsoTrackTracksCollection')
)



