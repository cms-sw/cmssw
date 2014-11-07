import FWCore.ParameterSet.Config as cms
from copy import deepcopy

SUSY_HLT_Mu_HT_MET_SingleLepton = cms.EDAnalyzer('SUSY_HLT_SingleLepton',
                                                 electronCollection = cms.InputTag(''),
                                                 muonCollection = cms.InputTag('muons'),
                                                 pfMetCollection = cms.InputTag('pfMet'),
                                                 pfJetCollection = cms.InputTag('ak4PFJets'),
                                                 jetTagCollection = cms.InputTag(''),

                                                 vertexCollection = cms.InputTag('goodOfflinePrimaryVertices'),
                                                 conversionCollection = cms.InputTag(''),
                                                 beamSpot = cms.InputTag(''),

                                                 leptonFilter = cms.InputTag('hltL3crVVVVLIsoL1sMu5L1f0L2f0QL3f15QL3crVVVVLIsoRhoFiltered1p0IterTrk02','','reHLT'),
                                                 hltHt = cms.InputTag('hltPFHT'),
                                                 hltMet = cms.InputTag('hltPFMETProducer'),
                                                 hltJets = cms.InputTag(''),
                                                 hltJetTags = cms.InputTag(''),
                                                 
                                                 triggerResults = cms.InputTag('TriggerResults','','reHLT'),
                                                 trigSummary = cms.InputTag('hltTriggerSummaryAOD','','reHLT'),

                                                 hltProcess = cms.string('reHLT'),

                                                 triggerPath = cms.string('HLT_Mu15_IterTrk02_IsoVVVL_PFHT400_PFMET70_v1'),
                                                 triggerPathAuxiliary = cms.string('HLT_IsoMu24_IterTrk02_v1'),

                                                 jetPtCut = cms.untracked.double(40.0),
                                                 jetEtaCut = cms.untracked.double(3.0),
                                                 metCut = cms.untracked.double(150.0),

                                                 leptonPtThreshold = cms.untracked.double(15.0),
                                                 htThreshold = cms.untracked.double(400.0),
                                                 metThreshold = cms.untracked.double(70.0),
                                                 csvThreshold = cms.untracked.double(-1.0)
                                                 )


SUSY_HLT_Mu_HT_MET_SingleLepton_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                                subDirs = cms.untracked.vstring('HLT/SUSYBSM/HLT_Mu15_IterTrk02_IsoVVVL_PFHT400_PFMET70_v1'),
                                                                efficiency = cms.vstring(
        "leptonPtTurnOn_eff ';Offline muon p_{T} [GeV];#epsilon' leptonPtTurnOn_num leptonPtTurnOn_den"
        "leptonIsoTurnOn_eff ';Offline muon rel. iso.;#epsilon' leptonIsoTurnOn_num leptonIsoTurnOn_den"
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den",
        "pfMetTurnOn_eff ';Offline PF MET [GeV];#epsilon' pfMetTurnOn_num pfMetTurnOn_den",
        ),
                                                                resolution = cms.vstring('')
                                                                )

SUSY_HLT_Mu_HT_MET_SingleLepton_FASTSIM = deepcopy(SUSY_HLT_Mu_HT_MET_SingleLepton)

SUSY_HLT_Mu_HT_MET_SingleLepton_FASTSIM_POSTPROCESSING = deepcopy(SUSY_HLT_Mu_HT_MET_SingleLepton_POSTPROCESSING)
