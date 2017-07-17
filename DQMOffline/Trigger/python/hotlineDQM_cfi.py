import FWCore.ParameterSet.Config as cms

hotlineDQM_HT = cms.EDAnalyzer('HotlineDQM',
     photonCollection = cms.InputTag('photons'),
     muonCollection = cms.InputTag('muons'),
     caloJetCollection = cms.InputTag('ak4CaloJets'),
     pfMetCollection = cms.InputTag('pfMet'),
     caloMetCollection = cms.InputTag('caloMet'),

     triggerResults = cms.InputTag('TriggerResults','','HLT'),
     trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

     triggerPath = cms.string('HLT_HT2000_v'),
     triggerFilter = cms.InputTag('hltHT2000', '', 'HLT'),

     useHT = cms.bool(True)
)

hotlineDQM_HT_Tight = cms.EDAnalyzer('HotlineDQM',
     photonCollection = cms.InputTag('photons'),
     muonCollection = cms.InputTag('muons'),
     caloJetCollection = cms.InputTag('ak4CaloJets'),
     pfMetCollection = cms.InputTag('pfMet'),
     caloMetCollection = cms.InputTag('caloMet'),

     triggerResults = cms.InputTag('TriggerResults','','HLT'),
     trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

     triggerPath = cms.string('HLT_HT2500_v'),
     triggerFilter = cms.InputTag('hltHT2500', '', 'HLT'),

     useHT = cms.bool(True)
)

hotlineDQM_Photon = cms.EDAnalyzer('HotlineDQM',
     photonCollection = cms.InputTag('photons'),
     muonCollection = cms.InputTag('muons'),
     caloJetCollection = cms.InputTag('ak4CaloJets'),
     pfMetCollection = cms.InputTag('pfMet'),
     caloMetCollection = cms.InputTag('caloMet'),

     triggerResults = cms.InputTag('TriggerResults','','HLT'),
     trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

     triggerPath = cms.string('HLT_Photon500_v'),
     triggerFilter = cms.InputTag('hltEG500HEFilter', '', 'HLT'),

     usePhotons = cms.bool(True)
)

hotlineDQM_Photon_Tight = cms.EDAnalyzer('HotlineDQM',
     photonCollection = cms.InputTag('photons'),
     muonCollection = cms.InputTag('muons'),
     caloJetCollection = cms.InputTag('ak4CaloJets'),
     pfMetCollection = cms.InputTag('pfMet'),
     caloMetCollection = cms.InputTag('caloMet'),

     triggerResults = cms.InputTag('TriggerResults','','HLT'),
     trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

     triggerPath = cms.string('HLT_Photon600_v'),
     triggerFilter = cms.InputTag('hltEG600HEFilter', '', 'HLT'),

     usePhotons = cms.bool(True)
)

hotlineDQM_MET = cms.EDAnalyzer('HotlineDQM',
     photonCollection = cms.InputTag('photons'),
     muonCollection = cms.InputTag('muons'),
     caloJetCollection = cms.InputTag('ak4CaloJets'),
     pfMetCollection = cms.InputTag('pfMet'),
     caloMetCollection = cms.InputTag('caloMet'),

     triggerResults = cms.InputTag('TriggerResults','','HLT'),
     trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

     triggerPath = cms.string('HLT_MET600_v'),
     triggerFilter = cms.InputTag('hltMETClean590', '', 'HLT'),

     useMet = cms.bool(True)
)

hotlineDQM_MET_Tight = cms.EDAnalyzer('HotlineDQM',
     photonCollection = cms.InputTag('photons'),
     muonCollection = cms.InputTag('muons'),
     caloJetCollection = cms.InputTag('ak4CaloJets'),
     pfMetCollection = cms.InputTag('pfMet'),
     caloMetCollection = cms.InputTag('caloMet'),

     triggerResults = cms.InputTag('TriggerResults','','HLT'),
     trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

     triggerPath = cms.string('HLT_MET700_v'),
     triggerFilter = cms.InputTag('hltMETClean690', '', 'HLT'),

     useMet = cms.bool(True)
)

hotlineDQM_PFMET = cms.EDAnalyzer('HotlineDQM',
     photonCollection = cms.InputTag('photons'),
     muonCollection = cms.InputTag('muons'),
     caloJetCollection = cms.InputTag('ak4CaloJets'),
     pfMetCollection = cms.InputTag('pfMet'),
     caloMetCollection = cms.InputTag('caloMet'),

     triggerResults = cms.InputTag('TriggerResults','','HLT'),
     trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

     triggerPath = cms.string('HLT_PFMET500_v'),
     triggerFilter = cms.InputTag('hltPFMET500Filter', '', 'HLT'),

     usePFMet = cms.bool(True)
)

hotlineDQM_PFMET_Tight = cms.EDAnalyzer('HotlineDQM',
     photonCollection = cms.InputTag('photons'),
     muonCollection = cms.InputTag('muons'),
     caloJetCollection = cms.InputTag('ak4CaloJets'),
     pfMetCollection = cms.InputTag('pfMet'),
     caloMetCollection = cms.InputTag('caloMet'),

     triggerResults = cms.InputTag('TriggerResults','','HLT'),
     trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

     triggerPath = cms.string('HLT_PFMET600_v'),
     triggerFilter = cms.InputTag('hltPFMET600Filter', '', 'HLT'),

     usePFMet = cms.bool(True)
)

hotlineDQM_Muon = cms.EDAnalyzer('HotlineDQM',
     photonCollection = cms.InputTag('photons'),
     muonCollection = cms.InputTag('muons'),
     caloJetCollection = cms.InputTag('ak4CaloJets'),
     pfMetCollection = cms.InputTag('pfMet'),
     caloMetCollection = cms.InputTag('caloMet'),

     triggerResults = cms.InputTag('TriggerResults','','HLT'),
     trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

     triggerPath = cms.string('HLT_Mu300_v'),
     triggerFilter = cms.InputTag('hltL3fL1sMu16orMu25L1f0L2f16QL3Filtered300Q', '', 'HLT'),

     useMuons = cms.bool(True)
)

hotlineDQM_Muon_Tight = cms.EDAnalyzer('HotlineDQM',
     photonCollection = cms.InputTag('photons'),
     muonCollection = cms.InputTag('muons'),
     caloJetCollection = cms.InputTag('ak4CaloJets'),
     pfMetCollection = cms.InputTag('pfMet'),
     caloMetCollection = cms.InputTag('caloMet'),

     triggerResults = cms.InputTag('TriggerResults','','HLT'),
     trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

     triggerPath = cms.string('HLT_Mu350_v'),
     triggerFilter = cms.InputTag('hltL3fL1sMu16orMu25L1f0L2f16QL3Filtered350Q', '', 'HLT'),

     useMuons = cms.bool(True)
)

hotlineDQMSequence = cms.Sequence(hotlineDQM_HT*hotlineDQM_HT_Tight*hotlineDQM_Photon*hotlineDQM_Photon_Tight*hotlineDQM_Muon*hotlineDQM_Muon_Tight*hotlineDQM_MET*hotlineDQM_MET_Tight*hotlineDQM_PFMET*hotlineDQM_PFMET_Tight)
