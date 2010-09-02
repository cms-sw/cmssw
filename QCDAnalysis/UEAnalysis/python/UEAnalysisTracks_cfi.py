import FWCore.ParameterSet.Config as cms

# old ( < CMSSW_2_x) cuts:
#
# select tracks with
# pt > 0.89
# at least 8 valid hits
# distance of closest approach to the primary vertex <= 3.5
# z-difference to primary vertex <= 30
# new ( >= CMSSW_2_x) cuts (under investigation):
#
# select tracks with
# pt > 0.29
# at least 4 valid hits -> moved back to 8
# distance of closest approach to the primary vertex <= 3.5
# z-difference to primary vertex <= 30
selectTracks = cms.EDFilter("TrackSelector",
    src = cms.InputTag("generalTracks"),
    cut = cms.string('pt > 0.29 & numberOfValidHits > 7 & d0 <= 3.5 & dz <= 30')
)

allTracks = cms.EDProducer("ChargedCandidateProducer",
    src = cms.InputTag("selectTracks"),
    particleType = cms.string('pi+')
)

goodTracks = cms.EDFilter("CandSelector",
    filter = cms.bool(False),
    src = cms.InputTag("allTracks"),
    cut = cms.string('pt > 0.29')
)

UEAnalysisTracks = cms.Sequence(selectTracks*allTracks*goodTracks)

