import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

ZGammaPostProc = DQMEDHarvester("ZGammaplusJetsPostProcessor",
     subDir = cms.untracked.string("HLT/JME/ZGammaPlusJets"),
     IsMuonTrigger = cms.untracked.string("Dimuon+(AK8*|CaloJet|PFJet)[0-9]+"),
     IsPhotonTrigger = cms.untracked.string("Photon([0-9])+(AK8*|CaloJet|PFJet)[0-9]+")
)
