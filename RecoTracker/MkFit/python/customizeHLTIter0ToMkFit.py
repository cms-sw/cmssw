import FWCore.ParameterSet.Config as cms

import RecoTracker.MkFit.mkFitGeometryESProducer_cfi as mkFitGeometryESProducer_cfi
import RecoTracker.MkFit.mkFitHitConverter_cfi as mkFitHitConverter_cfi
import RecoTracker.MkFit.mkFitSeedConverter_cfi as mkFitSeedConverter_cfi
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

    process.hltIter0PFlowCkfTrackCandidatesMkFitHits = mkFitHitConverter_cfi.mkFitHitConverter.clone(
        pixelRecHits = "hltSiPixelRecHits",
        stripRphiRecHits = "hltSiStripRecHits:rphiRecHit",
        stripStereoRecHits = "hltSiStripRecHits:stereoRecHit",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
        minGoodStripCharge = dict(refToPSet_ = 'HLTSiStripClusterChargeCutLoose'),
    )
    process.hltIter0PFlowCkfTrackCandidatesMkFitSeeds = mkFitSeedConverter_cfi.mkFitSeedConverter.clone(
        hits = "hltIter0PFlowCkfTrackCandidatesMkFitHits",
        seeds = "hltIter0PFLowPixelSeedsFromPixelTracks",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
    )
    process.hltIter0PFlowCkfTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
        hits = "hltIter0PFlowCkfTrackCandidatesMkFitHits",
        seeds = "hltIter0PFlowCkfTrackCandidatesMkFitSeeds",
    )
    process.hltIter0PFlowCkfTrackCandidates = mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
        seeds = "hltIter0PFLowPixelSeedsFromPixelTracks",
        mkfitHits = "hltIter0PFlowCkfTrackCandidatesMkFitHits",
        mkfitSeeds = "hltIter0PFlowCkfTrackCandidatesMkFitSeeds",
        tracks = "hltIter0PFlowCkfTrackCandidatesMkFit",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
        propagatorAlong = ":PropagatorWithMaterialParabolicMf",
        propagatorOpposite = ":PropagatorWithMaterialParabolicMfOpposite",
    )

    process.HLTDoLocalStripSequence += process.hltSiStripRecHits
    process.HLTIterativeTrackingIteration0.replace(process.hltIter0PFlowCkfTrackCandidates,
                                                   process.hltIter0PFlowCkfTrackCandidatesMkFitHits+process.hltIter0PFlowCkfTrackCandidatesMkFitSeeds+process.hltIter0PFlowCkfTrackCandidatesMkFit+process.hltIter0PFlowCkfTrackCandidates)

    return process
