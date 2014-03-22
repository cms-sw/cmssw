import FWCore.ParameterSet.Config as cms

# Data-quality monitoring for EWK tau analyses
ewkTauDQM = cms.EDAnalyzer("EwkTauDQM",

    # name of DQM (root)directory in which all EWK tau histograms get stored                          
    dqmDirectory = cms.string("Physics/EwkTau"),

    # disable all warnings
    maxNumWarnings = cms.int32(0),                       

    # list of individual channels                           
    channels = cms.PSet(
        muTauChannel = cms.PSet(
            # name of DQM (sub)directory in which histograms for Z --> muon + tau-jet channel get stored
            dqmSubDirectory = cms.string("MuTauChannel"),

            # names of input collections
            triggerResultsSource = cms.InputTag("TriggerResults::HLT"),
            vertexSource = cms.InputTag("offlinePrimaryVertices"),
            beamSpotSource = cms.InputTag("offlineBeamSpot"),
            muonSource = cms.InputTag("muons"),
            tauJetSource = cms.InputTag("shrinkingConePFTauProducer"),
            caloMEtSource = cms.InputTag("met"),
            pfMEtSource = cms.InputTag("pfMet"),

            tauDiscrByLeadTrackFinding = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackFinding"),
            tauDiscrByLeadTrackPtCut = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackPtCut"),
            tauDiscrByTrackIso = cms.InputTag("shrinkingConePFTauDiscriminationByTrackIsolation"),
            tauDiscrByEcalIso = cms.InputTag("shrinkingConePFTauDiscriminationByECALIsolation"),
            tauDiscrAgainstMuons = cms.InputTag("shrinkingConePFTauDiscriminationAgainstMuon"),

            # high-level trigger paths
            # (at least one of the paths specified in the list is required to be passed)
            hltPaths = cms.vstring(
               #"HLT_Mu9"
            ),

            # event selection criteria
            muonEtaCut = cms.double(2.5),
            muonPtCut = cms.double(15.),
            muonTrackIsoCut = cms.double(2.),
            muonEcalIsoCut = cms.double(3.),
             muonCombIsoCut = cms.double(0.2),
            #muonIsoMode = cms.string("absoluteIso"),
            muonIsoMode = cms.string("relativeIso"),

            tauJetEtaCut = cms.double(2.5),
            tauJetPtCut = cms.double(15.),

            visMassCut = cms.double(10.),
            deltaRCut = cms.double(0.2)

        ),
        elecTauChannel = cms.PSet(
            # name of DQM (sub)directory in which histograms for Z --> electron + tau-jet channel get stored
            dqmSubDirectory = cms.string("ElecTauChannel"),

            # names of input collections
            triggerResultsSource = cms.InputTag("TriggerResults::HLT"),
            vertexSource = cms.InputTag("offlinePrimaryVertices"),
            beamSpotSource = cms.InputTag("offlineBeamSpot"),
            electronSource = cms.InputTag("gedGsfElectrons"),
            tauJetSource = cms.InputTag("shrinkingConePFTauProducer"),
            caloMEtSource = cms.InputTag("met"),
            pfMEtSource = cms.InputTag("pfMet"),

            tauDiscrByLeadTrackFinding = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackFinding"),
            tauDiscrByLeadTrackPtCut = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackPtCut"),
            tauDiscrByTrackIso = cms.InputTag("shrinkingConePFTauDiscriminationByTrackIsolation"),
            tauDiscrByEcalIso = cms.InputTag("shrinkingConePFTauDiscriminationByECALIsolation"),
            tauDiscrAgainstElectrons = cms.InputTag("shrinkingConePFTauDiscriminationAgainstElectron"),
            tauDiscrAgainstMuons = cms.InputTag("shrinkingConePFTauDiscriminationAgainstMuon"),

            # high-level trigger paths
            # (at least one of the paths specified in the list is required to be passed)
            hltPaths = cms.vstring(
                ##"HLT_Ele10_LW_L1R"
            ),

            # event selection criteria
            electronEtaCut = cms.double(2.5),
            electronPtCut = cms.double(15.),
            electronTrackIsoCut = cms.double(0.2),
            electronEcalIsoCut = cms.double(0.2),
            electronIsoMode = cms.string("relativeIso"),

            tauJetEtaCut = cms.double(2.5),
            tauJetPtCut = cms.double(15.),

            visMassCut = cms.double(10.)
        )
    )
)                         
