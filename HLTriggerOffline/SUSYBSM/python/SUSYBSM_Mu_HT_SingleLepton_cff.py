import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from copy import deepcopy

SUSY_HLT_Mu15_HT600_SingleLepton = cms.EDAnalyzer('SUSY_HLT_SingleLepton',
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

                                             triggerPath = cms.string('HLT_Mu15_IsoVVVL_PFHT600_v'),
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
                                             htThreshold = cms.untracked.double(750.0),
                                             metThreshold = cms.untracked.double(-1.0),
                                             csvThreshold = cms.untracked.double(-1.0)
                                             )

SUSY_HLT_Mu15_HT600_SingleLepton_POSTPROCESSING = DQMEDHarvester('DQMGenericClient',
                                                            subDirs = cms.untracked.vstring('HLT/SUSYBSM/HLT_Mu15_IsoVVVL_PFHT600_v'),
                                                            efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
        ),
                                                            resolution = cms.vstring('')
                                                            )

SUSY_HLT_Mu15_HT400_SingleLepton = cms.EDAnalyzer('SUSY_HLT_SingleLepton',
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

                                             triggerPath = cms.string('HLT_Mu15_IsoVVVL_PFHT400_v'),
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
                                             htThreshold = cms.untracked.double(500.0),
                                             metThreshold = cms.untracked.double(-1.0),
                                             csvThreshold = cms.untracked.double(-1.0)
                                             )

SUSY_HLT_Mu15_HT400_SingleLepton_POSTPROCESSING = DQMEDHarvester('DQMGenericClient',
                                                            subDirs = cms.untracked.vstring('HLT/SUSYBSM/HLT_Mu15_IsoVVVL_PFHT400_v'),
                                                            efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
        ),
                                                            resolution = cms.vstring('')
                                                            )

SUSY_HLT_Mu50_HT400_SingleLepton = cms.EDAnalyzer('SUSY_HLT_SingleLepton',
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

                                             triggerPath = cms.string('HLT_Mu50_IsoVVVL_PFHT400_v'),
                                             triggerPathAuxiliary = cms.string('HLT_IsoMu27_v'),
                                             triggerPathLeptonAuxiliary = cms.string('HLT_PFHT350_PFMET120_NoiseCleaned_v'),

                                             csvlCut = cms.untracked.double(0.244),
                                             csvmCut = cms.untracked.double(0.679),
                                             csvtCut = cms.untracked.double(0.898),

                                             jetPtCut = cms.untracked.double(30.0),
                                             jetEtaCut = cms.untracked.double(3.0),
                                             metCut = cms.untracked.double(250.0),
                                             htCut = cms.untracked.double(450.0),

                                             leptonPtThreshold = cms.untracked.double(55.0),
                                             htThreshold = cms.untracked.double(500.0),
                                             metThreshold = cms.untracked.double(-1.0),
                                             csvThreshold = cms.untracked.double(-1.0)
                                             )

SUSY_HLT_Mu50_HT400_SingleLepton_POSTPROCESSING = DQMEDHarvester('DQMGenericClient',
                                                            subDirs = cms.untracked.vstring('HLT/SUSYBSM/HLT_Mu50_IsoVVVL_PFHT400_v'),
                                                            efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
        ),
                                                            resolution = cms.vstring('')
                                                            )


SUSY_HLT_Mu_HT_SingleLepton = cms.Sequence( SUSY_HLT_Mu15_HT600_SingleLepton
                                             + SUSY_HLT_Mu15_HT400_SingleLepton
                                             + SUSY_HLT_Mu50_HT400_SingleLepton
)

SUSY_HLT_Mu_HT_SingleLepton_POSTPROCESSING = cms.Sequence( SUSY_HLT_Mu15_HT600_SingleLepton_POSTPROCESSING
                                                            + SUSY_HLT_Mu15_HT400_SingleLepton_POSTPROCESSING
                                                            + SUSY_HLT_Mu50_HT400_SingleLepton_POSTPROCESSING

)
