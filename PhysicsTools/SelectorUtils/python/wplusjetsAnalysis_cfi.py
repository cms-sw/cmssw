import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector as pvSel
from PhysicsTools.SelectorUtils.jetIDSelector_cfi import jetIDSelector
from PhysicsTools.SelectorUtils.pfJetIDSelector_cfi import pfJetIDSelector



wplusjetsAnalysis = cms.PSet(
    # Primary vertex
    pvSelector = cms.PSet( pvSel.clone() ),
    # input parameter sets
    muonSrc = cms.InputTag('selectedPatMuons'),
    electronSrc = cms.InputTag('selectedPatElectrons'),
    jetSrc = cms.InputTag('selectedPatJets'),
    metSrc = cms.InputTag('patMETs'),
    trigSrc = cms.InputTag('patTriggerEvent'),
    muTrig = cms.string('HLT_Mu9'),
    eleTrig = cms.string('HLT_Ele15_LW_L1R'),
    # tight muons
    muonIdTight = cms.PSet(
        version = cms.string('SUMMER08'),
        Chi2 = cms.double(10.0),
        D0 = cms.double(0.2),
        NHits = cms.int32(11),
        ECalVeto = cms.double(4.0),
        HCalVeto = cms.double(6.0),
        RelIso = cms.double(0.05),
        ),
    # tight electrons
    electronIdTight = cms.PSet(
        version = cms.string('SUMMER08'),
        D0 = cms.double(0.02),
        RelIso = cms.double( 0.1 )
        ),
    # loose muons
    muonIdLoose = cms.PSet(
        version = cms.string('SUMMER08'),
        Chi2 = cms.double(10.0),
        D0 = cms.double(0.2),
        NHits = cms.int32(11),
        ECalVeto = cms.double(4.0),
        HCalVeto = cms.double(6.0),
        RelIso = cms.double(0.2),
        cutsToIgnore = cms.vstring('Chi2', 'D0', 'NHits','ECalVeto','HCalVeto')
        ),
    # loose electrons
    electronIdLoose = cms.PSet(
        version = cms.string('SUMMER08'),
        D0 = cms.double(0.2),
        RelIso = cms.double( 0.2 ),
        cutsToIgnore = cms.vstring('D0')
        ),
    # loose jets
    jetIdLoose = jetIDSelector.clone(),
    pfjetIdLoose = pfJetIDSelector.clone(),
    # kinematic cuts
    minJets        = cms.int32( 1 ),
    muPlusJets     = cms.bool( True ),
    ePlusJets      = cms.bool( False ),
    muPtMin        = cms.double( 20.0 ),
    muEtaMax       = cms.double( 2.1 ),
    elePtMin       = cms.double( 20.0 ),
    eleEtaMax      = cms.double( 2.4 ),
    muPtMinLoose   = cms.double( 20.0 ),
    muEtaMaxLoose  = cms.double( 2.1 ),
    elePtMinLoose  = cms.double( 20.0 ),
    eleEtaMaxLoose = cms.double( 2.4 ),    
    jetPtMin       = cms.double( 15.0 ),
    jetEtaMax      = cms.double( 3.0 ),
    jetScale       = cms.double( 1.0 ),
    metMin         = cms.double( 0.0 )
)
