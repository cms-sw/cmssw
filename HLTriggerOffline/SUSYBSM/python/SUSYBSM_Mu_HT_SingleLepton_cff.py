import FWCore.ParameterSet.Config as cms
from copy import deepcopy

SUSY_HLT_Mu_HT_SingleLepton = cms.EDAnalyzer('SUSY_HLT_SingleLepton',
                                             electronCollection = cms.InputTag(''),
                                             muonCollection = cms.InputTag('muons'),
                                             pfMetCollection = cms.InputTag('pfMet'),
                                             pfJetCollection = cms.InputTag('ak4PFJets'),
                                             jetTagCollection = cms.InputTag(''),

                                             vertexCollection = cms.InputTag('goodOfflinePrimaryVertices'),
                                             conversionCollection = cms.InputTag(''),
                                             beamSpot = cms.InputTag(''),

                                             leptonFilter = cms.InputTag('hltL3crIsoL1sMu5L1f0L2f3QL3f15QL3crIsoRhoFiltered0p15IterTrk02','','reHLT'),
                                             hltHt = cms.InputTag('hltPFHT','','reHLT'),
                                             hltMet = cms.InputTag(''),
                                             hltJets = cms.InputTag(''),
                                             hltJetTags = cms.InputTag(''),

                                             triggerResults = cms.InputTag('TriggerResults','','reHLT'),
                                             trigSummary = cms.InputTag('hltTriggerSummaryAOD','','reHLT'),

                                             hltProcess = cms.string('reHLT'),

                                             triggerPath = cms.string('HLT_Mu15_IsoVVVL_PFHT600'),
                                             triggerPathAuxiliary = cms.string('HLT_IsoMu24_v'),
                                             triggerPathLeptonAuxiliary = cms.string('HLT_PFHT350_PFMET120_NoiseCleaned_v'),

                                             jetPtCut = cms.untracked.double(40.0),
                                             jetEtaCut = cms.untracked.double(3.0),
                                             metCut = cms.untracked.double(250.0),
                                             htCut = cms.untracked.double(450.0),

                                             leptonPtThreshold = cms.untracked.double(25.0),
                                             htThreshold = cms.untracked.double(750.0),
                                             metThreshold = cms.untracked.double(-1.0),
                                             csvThreshold = cms.untracked.double(-1.0)
                                             )

SUSY_HLT_Mu_HT_SingleLepton_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                            subDirs = cms.untracked.vstring('HLT/SUSYBSM/HLT_Mu15_IsoVVVL_PFHT600'),
                                                            efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
        ),
                                                            resolution = cms.vstring('')
                                                            )

SUSY_HLT_Mu_HT_SingleLepton_FASTSIM = deepcopy(SUSY_HLT_Mu_HT_SingleLepton)

SUSY_HLT_Mu_HT_SingleLepton_FASTSIM_POSTPROCESSING = deepcopy(SUSY_HLT_Mu_HT_SingleLepton_POSTPROCESSING)
