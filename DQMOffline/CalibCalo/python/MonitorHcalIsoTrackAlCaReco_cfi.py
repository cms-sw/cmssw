import FWCore.ParameterSet.Config as cms

MonitorHcalIsoTrackAlCaReco = cms.EDAnalyzer("DQMHcalIsoTrackAlCaReco",
                                             folderName=cms.string("AlCaReco/HcalIsoTrack"),
                                             saveToFile=cms.bool(False),
                                             outputRootFileName=cms.string("HcalIsoTrackAlCaRecoMon.root"),
                                             TriggerLabel=cms.InputTag('hltTriggerSummaryAOD'),
                                             L1FilterLabel=cms.string('hltL1sJet68'),
                                             HltFilterLabels=cms.vstring('hltIsolPixelTrackL3FilterHB','hltIsolPixelTrackL3FilterHE','hltIsolPixelTrackL2FilterHB','hltIsolPixelTrackL2FilterHE','hltEcalIsolPixelTrackL2FilterHB','hltEcalIsolPixelTrackL2FilterHE'),
                                             TypeFilter=cms.vint32(2,2,0,0,1,1),
                                             TracksLabel=cms.InputTag('IsoProd:HcalIsolatedTrackCollection'),
                                             pThrL3=cms.untracked.double(0),
)
