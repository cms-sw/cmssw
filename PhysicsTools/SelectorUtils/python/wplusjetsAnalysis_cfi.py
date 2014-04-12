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
    jetClonesSrc = cms.InputTag('myClones'),
    metSrc = cms.InputTag('patMETs'),
    trigSrc = cms.InputTag('patTriggerEvent'),
    muTrig = cms.string('HLT_Mu9'),
    eleTrig = cms.string('HLT_Ele15_LW_L1R'),
    # tight muons
    muonIdTight = cms.PSet(
        version = cms.string('SPRING10'),
        Chi2 = cms.double(10.0),
        D0 = cms.double(0.02),
        ED0 = cms.double(999.0),
        SD0 = cms.double(999.0),
        NHits = cms.int32(11),
        NValMuHits = cms.int32(0),
        ECalVeto = cms.double(999.0),
        HCalVeto = cms.double(999.0),
        RelIso = cms.double(0.05),
        cutsToIgnore = cms.vstring('ED0', 'SD0', 'ECalVeto', 'HCalVeto'),
        RecalcFromBeamSpot = cms.bool(False),
        beamLineSrc = cms.InputTag("offlineBeamSpot")
        ),
    # tight electrons
    electronIdTight = cms.PSet(
        version = cms.string('FIRSTDATA'),
        D0 = cms.double(999.0),
        ED0 = cms.double(999.0),
        SD0 = cms.double(3.0),
        RelIso = cms.double( 0.1 ),
        cutsToIgnore = cms.vstring('D0', 'ED0')
        ),
    # loose muons
    muonIdLoose = cms.PSet(
        version = cms.string('SPRING10'),
        Chi2 = cms.double(999.0),
        D0 = cms.double(999.0),
        ED0 = cms.double(999.0),
        SD0 = cms.double(999.0),
        NHits = cms.int32(-1),
        NValMuHits = cms.int32(-1),
        ECalVeto = cms.double(999.0),
        HCalVeto = cms.double(999.0),
        RelIso = cms.double(0.2),
        cutsToIgnore = cms.vstring('Chi2', 'D0', 'ED0', 'SD0', 'NHits','NValMuHits','ECalVeto','HCalVeto'),
        RecalcFromBeamSpot = cms.bool(False),
        beamLineSrc = cms.InputTag("offlineBeamSpot")
        ),
    # loose electrons
    electronIdLoose = cms.PSet(
        version = cms.string('FIRSTDATA'),
        D0 = cms.double(999.0),
        ED0 = cms.double(999.0),
        SD0 = cms.double(999.0),
        RelIso = cms.double( 0.2 ),
        cutsToIgnore = cms.vstring( 'D0', 'ED0', 'SD0')
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
    eleEtMin       = cms.double( 20.0 ),
    eleEtaMax      = cms.double( 2.4 ),
    muPtMinLoose   = cms.double( 10.0 ),
    muEtaMaxLoose  = cms.double( 2.5 ),
    eleEtMinLoose  = cms.double( 15.0 ),
    eleEtaMaxLoose = cms.double( 2.5 ),    
    jetPtMin       = cms.double( 30.0 ),
    jetEtaMax      = cms.double( 2.4 ),
    jetScale       = cms.double( 1.0 ),
    metMin         = cms.double( 0.0 ),
    muJetDR        = cms.double( 0.3 ),
    useJetClones   = cms.bool(False),
    eleJetDR       = cms.double( 0.3 ),
    rawJetPtCut    = cms.double( 0.0 )
)
