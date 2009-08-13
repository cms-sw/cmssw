import FWCore.ParameterSet.Config as cms

higgsToTauTauElectronTauFilter = cms.EDFilter("HiggsToTauTauElectronTauSkim",
    # Collection to be accessed
    DebugHiggsToTauTauElectronTauSkim = cms.bool(False),
    HLTResultsCollection = cms.InputTag("TriggerResults::HLT8E29"),
    HLTEventCollection = cms.InputTag("hltTriggerSummaryAOD::HLT8E29"),
    #HLTFilterCollections = cms.vstring('hltL1IsoSingleElectronTrackIsolFilter'),
    HLTFilterCollections = cms.vstring('hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter'),
    #HLTElectronBits =  cms.vstring('HLT_IsoEle15_L1I'),
    HLTElectronBits =  cms.vstring('HLT_Ele10_LW_EleId_L1R'),
    minDRFromElectron = cms.double(0.5),
    jetEtaMin = cms.double(-2.6),
    jetEtaMax = cms.double(2.6),
    minNumberOfJets = cms.int32(1),
    minNumberOfElectrons = cms.int32(1),
    electronEtMin = cms.double(10.),
    electronEtaMax = cms.double(2.6),
    jetEtMin = cms.double(20.0),
    JetTagCollection = cms.InputTag("iterativeCone5CaloJets"),
    ElectronTagCollection = cms.InputTag("gsfElectrons"),
    #ElectronIdTagCollection = cms.InputTag("eidLoose")
    #ElectronIdTagCollection = cms.InputTag("eidRobustLoose") 
    #ElectronIdTagCollection = cms.InputTag("eidRobustTight")
    ElectronIdTagCollection = cms.InputTag("eidTight")
)
