import FWCore.ParameterSet.Config as cms

#------------------------------------------
# parameters for the CSCSkim module
#------------------------------------------
cscSkim = cms.EDFilter(
    "CSCSkim",
    typeOfSkim         = cms.untracked.int32(1),
    rootFileName       = cms.untracked.string('outputDummy.root'),
    histogramFileName  = cms.untracked.string('CSCSkim_histos.root'),
    nLayersWithHitsMinimum  = cms.untracked.int32(3),
    minimumHitChambers      = cms.untracked.int32(3),
    minimumSegments         = cms.untracked.int32(2),
    demandChambersBothSides = cms.untracked.bool(False),
    makeHistograms          = cms.untracked.bool(False),
    whichEndcap = cms.untracked.int32(2),
    whichStation = cms.untracked.int32(3),
    whichRing = cms.untracked.int32(2),
    whichChamber = cms.untracked.int32(24),
#
    cscRecHitTag  = cms.InputTag("csc2DRecHits"),
    cscSegmentTag = cms.InputTag("cscSegments"),
    SAMuonTag     = cms.InputTag("cosmicMuonsEndCapsOnly"),
    GLBMuonTag    = cms.InputTag("muonsEndCapsOnly"),
    trackTag      = cms.InputTag("ctfWithMaterialTracksP5")
)

#
