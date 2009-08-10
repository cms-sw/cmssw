import FWCore.ParameterSet.Config as cms

higgsToTauTauMuonTauFilter = cms.EDFilter("HiggsToTauTauMuonTauSkim",
    # Collection to be accessed
    DebugHiggsToTauTauMuonTauSkim = cms.bool(False),
    HLTResultsCollection = cms.InputTag("TriggerResults::HLT"),
    HLTEventCollection = cms.InputTag("hltTriggerSummaryAOD"),
    HLTFilterCollections = cms.vstring('hltSingleMuIsoL3IsoFiltered15',
	                               'hltSingleMuNoIsoL3PreFiltered11'),
    HLTMuonBits =  cms.vstring('HLT_Mu15','HLT_IsoMu11'),     
    minDRFromMuon = cms.double(0.5),
    jetEtaMin = cms.double(-2.6),
    jetEtaMax = cms.double(2.6),
    minNumberOfJets = cms.int32(1),
    jetEtMin = cms.double(20.0),
    JetTagCollection = cms.InputTag("iterativeCone5CaloJets")
)
