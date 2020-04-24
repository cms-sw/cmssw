import FWCore.ParameterSet.Config as cms

from math import pi

B2GDQM = cms.EDAnalyzer(
    "B2GDQM",

    #Trigger Results
    triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT"),

    PFJetCorService          = cms.string("ak4PFL1FastL2L3"),

    jetLabels = cms.VInputTag(
        'ak4PFJets',
        'ak4PFJetsCHS',
        'ak8PFJetsCHS',
        'ak8PFJetsCHSSoftDrop',
        'cmsTopTagPFJetsCHS'
        ),
    jetPtMins = cms.vdouble(
        50.,
        50.,
        50.,
        50.,
        100.
        ),
    pfMETCollection          = cms.InputTag("pfMet"),

    cmsTagLabel = cms.InputTag("cmsTopTagPFJetsCHS"),
    muonSrc = cms.InputTag("muons"),
    electronSrc = cms.InputTag("gedGsfElectrons"),

    allHadPtCut = cms.double(400.0),
    allHadRapidityCut = cms.double(1.0),
    allHadDeltaPhiCut = cms.double( pi / 2.0),

    muonSelect = cms.string("pt > 45.0 & abs(eta)<2.1 & isGlobalMuon & abs(globalTrack.d0)<1 & abs(globalTrack.dz)<20"),
    semiMu_HadJetPtCut = cms.double(400.0),
    semiMu_LepJetPtCut = cms.double(30.0),
    semiMu_dphiHadCut = cms.double( pi / 2.0),
    semiMu_dRMin = cms.double( 0.5 ),
    semiMu_ptRel = cms.double( 25.0 ),

    elecSelect = cms.string("pt > 45.0 & abs(eta)<2.5 & abs(gsfTrack.d0)<1 & abs(gsfTrack.dz)<20"),
    semiE_HadJetPtCut = cms.double(400.0),
    semiE_LepJetPtCut = cms.double(30.0),
    semiE_dphiHadCut = cms.double( pi / 2.0),
    semiE_dRMin = cms.double( 0.5 ),
    semiE_ptRel = cms.double( 25.0 )    


)
