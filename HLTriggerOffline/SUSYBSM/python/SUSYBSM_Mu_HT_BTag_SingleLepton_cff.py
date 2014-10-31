import FWCore.ParameterSet.Config as cms

SUSY_HLT_Mu_HT_BTag_SingleLepton = cms.EDAnalyzer("SUSY_HLT_SingleLepton",
                                                  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'','reHLT'),
                                                  ElectronCollection = cms.InputTag(""),
                                                  MuonCollection = cms.InputTag("muons"),
                                                  pfMETCollection = cms.InputTag(""),
                                                  pfJetCollection = cms.InputTag("ak4PFJets"),
                                                  jetTagCollection = cms.InputTag("pfCombinedSecondaryVertexBJetTags"),
                                                  TriggerResults = cms.InputTag('TriggerResults','','reHLT'),
                                                  HLTProcess = cms.string("reHLT"),
                                                  TriggerPath = cms.string('HLT_Mu15_IterTrk02_IsoVVVL_BTagtop8CSV07_PFHT400_v1'),
                                                  TriggerPathAuxiliary = cms.string('HLT_IsoMu24_IterTrk02_v1'),
                                                  TriggerFilter = cms.InputTag('hltL3crVVVVLIsoL1sMu5L1f0L2f0QL3f15QL3crVVVVLIsoRhoFiltered1p0IterTrk02','','reHLT'),
                                                  JetPtCut = cms.untracked.double(40.0),
                                                  JetEtaCut = cms.untracked.double(3.0),
                                                  LeptonPtCut = cms.untracked.double(40.0),
                                                  LeptonPtPlateau = cms.untracked.double(25.0),
                                                  HtPlateau = cms.untracked.double(500.0),
                                                  MetPlateau = cms.untracked.double(-1.0),
                                                  CsvPlateau = cms.untracked.double(0.898)
                                                  )

SUSY_HLT_Mu_HT_BTag_SingleLepton_FASTSIM = cms.EDAnalyzer("SUSY_HLT_SingleLepton",
                                                          trigSummary = cms.InputTag("hltTriggerSummaryAOD",'','reHLT'),
                                                          ElectronCollection = cms.InputTag(""),
                                                          MuonCollection = cms.InputTag("muons"),
                                                          pfMETCollection = cms.InputTag(""),
                                                          pfJetCollection = cms.InputTag("ak4PFJets"),
                                                          jetTagCollection = cms.InputTag("pfCombinedSecondaryVertexBJetTags"),
                                                          TriggerResults = cms.InputTag('TriggerResults','','reHLT'),
                                                          HLTProcess = cms.string("reHLT"),
                                                          TriggerPath = cms.string('HLT_Mu15_IterTrk02_IsoVVVL_BTagtop8CSV07_PFHT400_v1'),
                                                          TriggerPathAuxiliary = cms.string('HLT_IsoMu24_IterTrk02_v1'),
                                                          TriggerFilter = cms.InputTag('hltL3crVVVVLIsoL1sMu5L1f0L2f0QL3f15QL3crVVVVLIsoRhoFiltered1p0IterTrk02','','reHLT'),
                                                          JetPtCut = cms.untracked.double(40.0),
                                                          JetEtaCut = cms.untracked.double(3.0),
                                                          LeptonPtCut = cms.untracked.double(40.0),
                                                          LeptonPtPlateau = cms.untracked.double(25.0),
                                                          HtPlateau = cms.untracked.double(500.0),
                                                          MetPlateau = cms.untracked.double(-1.0),
                                                          CsvPlateau = cms.untracked.double(0.898)
                                                          )


SUSY_HLT_Mu_HT_BTag_SingleLepton_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
                                                                 subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu15_IterTrk02_IsoVVVL_BTagtop8CSV07_PFHT400_v1"),
                                                                 efficiency = cms.vstring(
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den",
        "CSVTurnOn_eff ';Offline b-tag requirements;#epsilon' CSVTurnOn_num CSVTurnOn_den",
        "leptonTurnOn_eff ';Offline muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den"
        ),
                                                                 resolution = cms.vstring("")
                                                                 )


SUSY_HLT_Mu_HT_BTag_SingleLepton_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
                                                                         subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu15_IterTrk02_IsoVVVL_BTagtop8CSV07_PFHT400_v1"),
                                                                         efficiency = cms.vstring(
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den",
        "CSVTurnOn_eff ';Offline b-tag requirements;#epsilon' CSVTurnOn_num CSVTurnOn_den",
        "leptonTurnOn_eff ';Offline muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den"
        ),
                                                                         resolution = cms.vstring("")
                                                                         )



