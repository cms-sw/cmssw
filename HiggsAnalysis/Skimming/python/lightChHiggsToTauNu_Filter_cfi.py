import FWCore.ParameterSet.Config as cms



lightChHiggsToTauNuFilter = cms.EDFilter("LightChHiggsToTauNuSkim",


    # Collection to be accessed
    jetsTag       = cms.InputTag("iterativeCone5CaloJets"),
    muonsTag      = cms.InputTag("muons"),
    electronsTag  = cms.InputTag("gsfElectrons"),
#    electronIdTag = cms.InputTag("eidTight"),
    electronIdTag = cms.InputTag("none"),

    # Lepton Thresholds
    leptonPtMin   = cms.double(15),
    leptonEtaMin  = cms.double(-2.4),
    leptonEtaMax  = cms.double(2.4),

    # jet thresolds
    minNumbOfJets = cms.int32(2),
    jetPtMin      = cms.double(15),
    jetEtaMin     = cms.double(-2.4),
    jetEtaMax     = cms.double(2.4),

   
    # trigger 
    triggerEventTag    = cms.InputTag("hltTriggerSummaryAOD"),
    hltFilters         = cms.vstring("hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter","hltSingleMu9L3Filtered9" ),

    drHLT              = cms.double(0.3), # dr between triggered lepton and jets
    drHLTMatch         = cms.double(0.4), # dr to match reco leptons and trigger objectd
       
)

