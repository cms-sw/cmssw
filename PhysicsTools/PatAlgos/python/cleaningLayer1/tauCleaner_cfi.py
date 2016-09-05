import FWCore.ParameterSet.Config as cms

## string added as general entry for modeul definition here and use in tauTools.py
preselection = cms.string(
    'tauID("decayModeFinding") > 0.5 &'
    ' tauID("byLooseCombinedIsolationDeltaBetaCorr3Hits") > 0.5 &'
    ' tauID("againstMuonTight3") > 0.5 &'
    ' tauID("againstElectronVLooseMVA6") > 0.5'
    )

cleanPatTaus = cms.EDProducer("PATTauCleaner",
    src = cms.InputTag("selectedPatTaus"),

    # preselection (any string-based cut on pat::Tau)
    preselection =  cms.string(
    'tauID("decayModeFinding") > 0.5 &'
    ' tauID("byLooseCombinedIsolationDeltaBetaCorr3Hits") > 0.5 &'
    ' tauID("againstMuonTight3") > 0.5 &'
    ' tauID("againstElectronVLooseMVA6") > 0.5'
    ),
                              
    # overlap checking configurables
    checkOverlaps = cms.PSet(
        muons = cms.PSet(
           src       = cms.InputTag("cleanPatMuons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.3),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOverlaps   = cms.bool(False), # overlaps don't cause the electron to be discared
        ),
        electrons = cms.PSet(
           src       = cms.InputTag("cleanPatElectrons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.3),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOverlaps   = cms.bool(False), # overlaps don't cause the electron to be discared
        ),
    ),

    # finalCut (any string-based cut on pat::Tau)
    finalCut = cms.string('pt > 18. & abs(eta) < 2.3'),
)
