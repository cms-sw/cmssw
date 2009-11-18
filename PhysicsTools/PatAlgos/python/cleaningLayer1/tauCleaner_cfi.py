import FWCore.ParameterSet.Config as cms

cleanLayer1Taus = cms.EDFilter("PATTauCleaner",
    src = cms.InputTag("selectedLayer1Taus"), 

    # preselection (any string-based cut on pat::Tau)
    preselection = cms.string(
        'tauID("leadingTrackFinding") > 0.5 &'
        ' tauID("leadingPionPtCut") > 0.5 &'
        ' tauID("byIsolationUsingLeadingPion") > 0.5 &'
        ' tauID("againstMuon") > 0.5 &'
        ' tauID("againstElectron") > 0.5 &'
        ' (signalTracks.size() = 1 | signalTracks.size() = 3)'
    ),

    # overlap checking configurables
    checkOverlaps = cms.PSet(
        muons = cms.PSet(
           src       = cms.InputTag("cleanLayer1Muons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.3),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOverlaps   = cms.bool(False), # overlaps don't cause the electron to be discared
        ),
        electrons = cms.PSet(
           src       = cms.InputTag("cleanLayer1Electrons"),
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
