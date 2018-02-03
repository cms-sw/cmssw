import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from copy import deepcopy

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SUSY_HLT_Mu_HT_Control_SingleLepton = DQMEDAnalyzer('SUSY_HLT_SingleLepton',
                                                     electronCollection = cms.InputTag(''),
                                                     muonCollection = cms.InputTag('muons'),
                                                     pfMetCollection = cms.InputTag('pfMet'),
                                                     pfJetCollection = cms.InputTag('ak4PFJets'),
                                                     jetTagCollection = cms.InputTag(''),

                                                     vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                                     conversionCollection = cms.InputTag(''),
                                                     beamSpot = cms.InputTag(''),

                                                     leptonFilter = cms.InputTag('hltL3MuVVVLIsoFIlter','','HLT'),
                                                     hltHt = cms.InputTag('hltPFHTJet30','','HLT'),
                                                     hltMet = cms.InputTag(''),
                                                     hltJets = cms.InputTag(''),
                                                     hltJetTags = cms.InputTag(''),

                                                     triggerResults = cms.InputTag('TriggerResults','','HLT'),
                                                     trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

                                                     hltProcess = cms.string('HLT'),

                                                     triggerPath = cms.string('HLT_Mu15_IsoVVVL_PFHT350_v'),
                                                     triggerPathAuxiliary = cms.string('HLT_IsoMu27_v'),
                                                     triggerPathLeptonAuxiliary = cms.string('HLT_PFHT350_PFMET120_NoiseCleaned_v'),

                                                     csvlCut = cms.untracked.double(0.244),
                                                     csvmCut = cms.untracked.double(0.679),
                                                     csvtCut = cms.untracked.double(0.898),

                                                     jetPtCut = cms.untracked.double(30.0),
                                                     jetEtaCut = cms.untracked.double(3.0),
                                                     metCut = cms.untracked.double(250.0),
                                                     htCut = cms.untracked.double(450.0),

                                                     leptonPtThreshold = cms.untracked.double(25.0),
                                                     htThreshold = cms.untracked.double(450.0),
                                                     metThreshold = cms.untracked.double(-1.0),
                                                     csvThreshold = cms.untracked.double(-1.0)
                                                     )

SUSYoHLToMuHToControlSingleLeptonPOSTPROCESSING = DQMEDHarvester('DQMGenericClient',
                                                                    subDirs = cms.untracked.vstring('HLT/SUSYBSM/HLT_Mu15_IsoVVVL_PFHT350_v'),
                                                                    efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
        ),
                                                                    resolution = cms.vstring('')
                                                                    )

