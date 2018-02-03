import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
MonitorHcalIsoTrackAlCaReco = DQMEDAnalyzer('DQMHcalIsoTrackAlCaReco',
                                             FolderName=cms.string("AlCaReco/HcalIsoTrack"),
                                             TriggerLabel=cms.InputTag('hltTriggerSummaryAOD'),
                                             L1FilterLabel=cms.vstring('L1SingleJet60'),
                                             HltFilterLabels=cms.vstring('hltIsolPixelTrackL3FilterHB','hltIsolPixelTrackL3FilterHE','hltIsolPixelTrackL2FilterHB','hltIsolPixelTrackL2FilterHE','hltEcalIsolPixelTrackL2FilterHB','hltEcalIsolPixelTrackL2FilterHE'),
                                             TypeFilter=cms.vint32(2,2,0,0,1,1),
                                             TracksLabel=cms.InputTag('IsoProd:HcalIsolatedTrackCollection'),
                                             pThrL3=cms.untracked.double(0),
)
