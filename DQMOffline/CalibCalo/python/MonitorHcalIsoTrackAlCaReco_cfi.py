import FWCore.ParameterSet.Config as cms

MonitorHcalIsoTrackAlCaReco = cms.EDAnalyzer("DQMHcalIsoTrackAlCaReco",
folderName=cms.string("AlCaReco/HcalIsoTrack"),
saveToFile=cms.bool(False),
outputRootFileName=cms.string("HcalIsoTrackAlCaRecoMon.root"),
hltTriggerEventLabel=cms.InputTag('hltTriggerSummaryAOD'),
hltL3FilterLabels=cms.vstring('hltIsolPixelTrackL3FilterHB1E31','hltIsolPixelTrackL3FilterHE1E31'),
alcarecoIsoTracksLabel=cms.InputTag('IsoProd:HcalIsolatedTrackCollection'),
recoTracksLabel=cms.InputTag('IsoProd:IsoTrackTracksCollection')
)



