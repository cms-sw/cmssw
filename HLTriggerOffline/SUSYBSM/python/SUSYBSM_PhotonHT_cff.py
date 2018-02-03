import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SUSY_HLT_PhotonHT = DQMEDAnalyzer('SUSY_HLT_PhotonHT',
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'),
  pfMETCollection = cms.InputTag("pfMet"),
  photonCollection = cms.InputTag("gedPhotons"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Photon90_CaloIdL_PFHT500_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilterPhoton = cms.InputTag('hltEG90CaloIdLHEFilter', '', 'HLT'),
  TriggerFilterHt = cms.InputTag('hltPFHT500Jet30', '', 'HLT'),
  ptThrOffline = cms.untracked.double( 100 ),
  htThrOffline = cms.untracked.double( 600 ),
)

SUSYoHLToPhotonHToPOSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Photon90_CaloIdL_PFHT500_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "photonPtTurnOn_eff 'Turn-on vs photon; E_{T} (GeV); #epsilon' photonTurnOn_num photonTurnOn_den",
       "htTurnOn_eff 'Turn-on vs H_{T}; H_{T} (GeV); #epsilon' pfHtTurnOn_num pfHtTurnOn_den",
    )
)
