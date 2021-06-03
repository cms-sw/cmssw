import FWCore.ParameterSet.Config as cms

import RecoTracker.MkFit.mkFitGeometryESProducer_cfi as mkFitGeometryESProducer_cfi
import RecoTracker.MkFit.mkFitSiPixelHitConverter_cfi as mkFitSiPixelHitConverter_cfi
import RecoTracker.MkFit.mkFitSiStripHitConverter_cfi as mkFitSiStripHitConverter_cfi
import RecoTracker.MkFit.mkFitEventOfHitsProducer_cfi as mkFitEventOfHitsProducer_cfi
import RecoTracker.MkFit.mkFitSeedConverter_cfi as mkFitSeedConverter_cfi
import RecoTracker.MkFit.mkFitIterationConfigESProducer_cfi as mkFitIterationConfigESProducer_cfi
import RecoTracker.MkFit.mkFitProducer_cfi as mkFitProducer_cfi
import RecoTracker.MkFit.mkFitOutputConverter_cfi as mkFitOutputConverter_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi as SiStripRecHitConverter_cfi

def customizeHLTIter0ToMkFit(process):
    # mkFit needs all clusters, so switch off the on-demand mode
    process.hltSiStripRawToClustersFacility.onDemand = False

    process.hltSiStripRecHits = SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
        ClusterProducer = "hltSiStripRawToClustersFacility",
        StripCPE = "hltESPStripCPEfromTrackAngle:hltESPStripCPEfromTrackAngle",
        doMatching = False,
    )

    # Use fourth hit if one is available
    process.hltIter0PFLowPixelSeedsFromPixelTracks.includeFourthHit = cms.bool(True)

    process.hltMkFitGeometryESProducer = mkFitGeometryESProducer_cfi.mkFitGeometryESProducer.clone()

    process.hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits = mkFitSiPixelHitConverter_cfi.mkFitSiPixelHitConverter.clone(
        hits = "hltSiPixelRecHits",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
    )
    process.hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits = mkFitSiStripHitConverter_cfi.mkFitSiStripHitConverter.clone(
        rphiHits = "hltSiStripRecHits:rphiRecHit",
        stereoHits = "hltSiStripRecHits:stereoRecHit",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
        minGoodStripCharge = dict(refToPSet_ = 'HLTSiStripClusterChargeCutLoose'),
    )
    process.hltIter0PFlowCkfTrackCandidatesMkFitEventOfHits = mkFitEventOfHitsProducer_cfi.mkFitEventOfHitsProducer.clone(
        pixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits",
        stripHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits",
    )
    process.hltIter0PFlowCkfTrackCandidatesMkFitSeeds = mkFitSeedConverter_cfi.mkFitSeedConverter.clone(
        seeds = "hltIter0PFLowPixelSeedsFromPixelTracks",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
    )
    process.hltIter0PFlowTrackCandidatesMkFitConfig = mkFitIterationConfigESProducer_cfi.mkFitIterationConfigESProducer.clone(
        ComponentName = 'hltIter0PFlowTrackCandidatesMkFitConfig',
        config = 'RecoTracker/MkFit/data/mkfit-phase1-initialStep.json',
    )
    process.hltIter0PFlowCkfTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
        pixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits",
        stripHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits",
        eventOfHits = "hltIter0PFlowCkfTrackCandidatesMkFitEventOfHits",
        seeds = "hltIter0PFlowCkfTrackCandidatesMkFitSeeds",
        config = ('', 'hltIter0PFlowTrackCandidatesMkFitConfig'),
        minGoodStripCharge = dict(refToPSet_ = 'HLTSiStripClusterChargeCutLoose'),
    )
    process.hltIter0PFlowCkfTrackCandidates = mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
        seeds = "hltIter0PFLowPixelSeedsFromPixelTracks",
        mkFitEventOfHits = "hltIter0PFlowCkfTrackCandidatesMkFitEventOfHits",
        mkFitPixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits",
        mkFitStripHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits",
        mkFitSeeds = "hltIter0PFlowCkfTrackCandidatesMkFitSeeds",
        tracks = "hltIter0PFlowCkfTrackCandidatesMkFit",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
        propagatorAlong = ":PropagatorWithMaterialParabolicMf",
        propagatorOpposite = ":PropagatorWithMaterialParabolicMfOpposite",
    )

    process.HLTDoLocalStripSequence += process.hltSiStripRecHits
    process.HLTIterativeTrackingIteration0.replace(process.hltIter0PFlowCkfTrackCandidates,
                                                   process.hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits +
                                                   process.hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits +
                                                   process.hltIter0PFlowCkfTrackCandidatesMkFitEventOfHits +
                                                   process.hltIter0PFlowCkfTrackCandidatesMkFitSeeds +
                                                   process.hltIter0PFlowCkfTrackCandidatesMkFit +
                                                   process.hltIter0PFlowCkfTrackCandidates)

    return process
