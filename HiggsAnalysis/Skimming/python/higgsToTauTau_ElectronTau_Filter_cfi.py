import FWCore.ParameterSet.Config as cms

higgsToTauTauElectronTauFilter = cms.EDFilter("HiggsToTauTauElectronTauSkim",
    # Collection to be accessed
    DebugHiggsToTauTauElectronTauSkim = cms.bool(False),
    HLTResultsCollection = cms.InputTag("TriggerResults::HLT"),
    HLTEventCollection = cms.InputTag("hltTriggerSummaryAOD"),
    HLTFilterCollections = cms.vstring('hltL1IsoSingleElectronTrackIsolFilter'),
    HLTElectronBits =  cms.vstring('HLT_IsoEle15_L1I'),
    minDRFromElectron = cms.double(0.5),
    jetEtaMin = cms.double(-2.6),
    jetEtaMax = cms.double(2.6),
    minNumberOfJets = cms.int32(1),
    minNumberOfElectrons = cms.int32(1),
    jetEtMin = cms.double(20.0),
    JetTagCollection = cms.InputTag("iterativeCone5CaloJets"),
    ElectronTagCollection = cms.InputTag("pixelMatchGsfElectrons"),
    ElectronIdTagCollection = cms.InputTag("eidRobustTight") 
)
