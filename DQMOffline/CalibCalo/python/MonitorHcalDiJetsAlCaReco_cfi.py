import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
MonitorHcalDiJetsAlCaReco = DQMEDAnalyzer('DQMHcalDiJetsAlCaReco',
FolderName=cms.untracked.string("AlCaReco/HcalDiJets"),
SaveToFile=cms.untracked.bool(False),
FileName=cms.untracked.string("HcalDiJetsAlCaRecoMon.root"),
jetsInput = cms.InputTag('DiJProd:DiJetsBackToBackCollection'),
ecInput  = cms.InputTag('DiJProd:DiJetsEcalRecHitCollection'),
hbheInput = cms.InputTag('DiJProd:DiJetsHBHERecHitCollection'),
hfInput = cms.InputTag('DiJProd:DiJetsHFRecHitCollection'),
hoInput = cms.InputTag('DiJProd:DiJetsHORecHitCollection')
)



