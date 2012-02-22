import FWCore.ParameterSet.Config as cms

cleanPatTaus = cms.EDProducer("PATTauCleaner",
    src = cms.InputTag("selectedPatTaus"), 

    # preselection (any string-based cut on pat::Tau)
    preselection = cms.string(
        'tauID("leadingTrackFinding") > 0.5 &'
        ' tauID("leadingPionPtCut") > 0.5 &'
        ' tauID("byIsolationUsingLeadingPion") > 0.5 &'
        ' tauID("againstMuon") > 0.5 &'
        ' tauID("againstElectron") > 0.5 &'
        ' (signalPFChargedHadrCands.size() = 1 | signalPFChargedHadrCands.size() = 3)'
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
    finalCut = cms.string(''),
)

