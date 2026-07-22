import FWCore.ParameterSet.Config as cms

hltPfTICL = cms.EDProducer("PFTICLProducer",
    mightGet = cms.optional.untracked.vstring,
    muonSrc = cms.InputTag("hltPhase2L3Muons"),
    pfMuonAlgoParameters = cms.PSet(
        cosmicRejectionDistance = cms.double(1),
        eventFactorForCosmics = cms.double(10),
        eventFractionForCleaning = cms.double(0.5),
        eventFractionForRejection = cms.double(0.8),
        maxDPtOPt = cms.double(1),
        metFactorForCleaning = cms.double(4),
        metFactorForFakes = cms.double(4),
        metFactorForHighEta = cms.double(25),
        metFactorForRejection = cms.double(4),
        metSignificanceForCleaning = cms.double(3),
        metSignificanceForRejection = cms.double(4),
        minEnergyForPunchThrough = cms.double(100),
        minMomentumForPunchThrough = cms.double(100),
        minPtForPostCleaning = cms.double(20),
        ptErrorScale = cms.double(8),
        ptFactorForHighEta = cms.double(2),
        punchThroughFactor = cms.double(3),
        punchThroughMETFactor = cms.double(4),
        trackQuality = cms.string('highPurity')
    ),
    ticlCandidateSrc = cms.InputTag("hltTiclCandidate"),
    timingQualityThreshold = cms.float(0.5),
    trackTimeErrorMap = cms.InputTag("hltTofPID","sigmat0"),
    trackTimeQualityMap = cms.InputTag("hltMtdTrackQualityMVA","mtdQualMVA"),
    trackTimeValueMap = cms.InputTag("hltTofPID","t0"),
    useMTDTiming = cms.bool(False),
    useTimingAverage = cms.bool(False)
)

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
mtd_at_hlt.toModify(hltPfTICL,
                    useMTDTiming = True)
