import FWCore.ParameterSet.Config as cms

# Data-quality monitoring for PF tau analyses
pfTauDQM = cms.EDAnalyzer("PFTauDQM",

    # name of DQM (root)directory in which all PFTau histograms get stored                          
    dqmDirectory = cms.string("PFTau"),

    # disable all warnings
    maxNumWarnings = cms.int32(0),                       

    # names of input collections
    triggerResultsSource = cms.InputTag("TriggerResults::HLT"),
    vertexSource = cms.InputTag("offlinePrimaryVertices"),
    tauJetSource = cms.InputTag("shrinkingConePFTauProducer"),
    hpsTauJetSource = cms.InputTag("hpsPFTauProducer"),                      

    tauDiscrByLeadTrackFinding = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackFinding"),
    tauDiscrByLeadTrackPtCut = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackPtCut"),
    tauDiscrByTrackIso = cms.InputTag("shrinkingConePFTauDiscriminationByTrackIsolation"),
    tauDiscrByEcalIso = cms.InputTag("shrinkingConePFTauDiscriminationByECALIsolation"),
    tauDiscrAgainstElectrons = cms.InputTag("shrinkingConePFTauDiscriminationAgainstElectron"),                      
    tauDiscrAgainstMuons = cms.InputTag("shrinkingConePFTauDiscriminationAgainstMuon"),
    tauDiscrTaNCWorkingPoint = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),
    tauDiscrTaNC = cms.InputTag("shrinkingConePFTauDiscriminationByTaNC"),
    tauDiscrHPSWorkingPoint = cms.InputTag("hpsPFTauDiscriminationByMediumIsolation"),                          

    # high-level trigger paths
    # (at least one of the paths specified in the list is required to be passed)
    hltPaths = cms.vstring(
        "HLT_Jet15U_HcalNoiseFiltered",
        "HLT_Jet30U",
        "HLT_Jet50U",                              
        "HLT_Jet70U"
    ),

    # event selection criteria
    tauJetPtCut = cms.double(10.),                      
    tauJetEtaCut = cms.double(2.5),
    tauJetLeadTrkDxyCut = cms.double(0.1),
    tauJetLeadTrkDzCut = cms.double(1.),
)                         
