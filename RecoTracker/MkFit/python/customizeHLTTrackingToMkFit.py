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

    # if any of the following objects does not exist, do not apply any customisation
    for objLabel in [
        'hltSiStripRawToClustersFacility',
        'HLTDoLocalStripSequence',
        'HLTIterativeTrackingIteration0',
        'hltIter0PFlowCkfTrackCandidates',
    ]:
        if not hasattr(process, objLabel):
            print(f'# WARNING: customizeHLTIter0ToMkFit failed (object with label "{objLabel}" not found) - no customisation applied !')
            return process

    # mkFit needs all clusters, so switch off the on-demand mode
    process.hltSiStripRawToClustersFacility = cms.EDProducer(
        "SiStripClusterizerFromRaw",
        ProductLabel = cms.InputTag( "rawDataCollector" ),
        ConditionsLabel = cms.string( "" ),
        onDemand = cms.bool( True ),
        DoAPVEmulatorCheck = cms.bool( False ),
        LegacyUnpacker = cms.bool( False ),
        HybridZeroSuppressed = cms.bool( False ),
        Clusterizer = cms.PSet( 
            ConditionsLabel = cms.string( "" ),
            MaxClusterSize = cms.uint32( 32 ), 
            ClusterThreshold = cms.double( 5.0 ),
            SeedThreshold = cms.double( 3.0 ),
            Algorithm = cms.string( "ThreeThresholdAlgorithm" ),
            ChannelThreshold = cms.double( 2.0 ),
            MaxAdjacentBad = cms.uint32( 0 ),
            setDetId = cms.bool( True ),
            MaxSequentialHoles = cms.uint32( 0 ),
            RemoveApvShots = cms.bool( True ),
            clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
            MaxSequentialBad = cms.uint32( 1 )
        ),
        Algorithms = cms.PSet( 
            Use10bitsTruncation = cms.bool( False ),
            CommonModeNoiseSubtractionMode = cms.string( "Median" ),
            useCMMeanMap = cms.bool( False ),
            TruncateInSuppressor = cms.bool( True ),
            doAPVRestore = cms.bool( False ),
            SiStripFedZeroSuppressionMode = cms.uint32( 4 ),
            PedestalSubtractionFedMode = cms.bool( True )
        )
    )
    process.hltSiStripRawToClustersFacility.onDemand = False
    process.hltSiStripRawToClustersFacility.Clusterizer.MaxClusterSize = 16

    process.hltSiStripRecHits = SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
        ClusterProducer = "hltSiStripRawToClustersFacility",
        StripCPE = "hltESPStripCPEfromTrackAngle:hltESPStripCPEfromTrackAngle",
        doMatching = False,
    )

    # Use fourth hit if one is available
    process.hltIter0PFLowPixelSeedsFromPixelTracks.includeFourthHit = cms.bool(True)

    process.load("RecoTracker.MkFit.mkFitGeometryESProducer_cfi")

    process.hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits = mkFitSiPixelHitConverter_cfi.mkFitSiPixelHitConverter.clone(
        hits = "hltSiPixelRecHits",
        clusters = "hltSiPixelClusters",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
    )
    process.hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits = mkFitSiStripHitConverter_cfi.mkFitSiStripHitConverter.clone(
        rphiHits = "hltSiStripRecHits:rphiRecHit",
        stereoHits = "hltSiStripRecHits:stereoRecHit",
        clusters = "hltSiStripRawToClustersFacility",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
        minGoodStripCharge = dict(refToPSet_ = 'HLTSiStripClusterChargeCutLoose'),
    )
    process.hltIter0PFlowCkfTrackCandidatesMkFitEventOfHits = mkFitEventOfHitsProducer_cfi.mkFitEventOfHitsProducer.clone(
        beamSpot  = "hltOnlineBeamSpot",
        pixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits",
        stripHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits",
    )
    process.hltIter0PFlowCkfTrackCandidatesMkFitSeeds = mkFitSeedConverter_cfi.mkFitSeedConverter.clone(
        seeds = "hltIter0PFLowPixelSeedsFromPixelTracks",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
    )
    process.hltIter0PFlowTrackCandidatesMkFitConfig = mkFitIterationConfigESProducer_cfi.mkFitIterationConfigESProducer.clone(
        ComponentName = 'hltIter0PFlowTrackCandidatesMkFitConfig',
        config = 'RecoTracker/MkFit/data/mkfit-phase1-hltiter0.json',
    )
    process.hltIter0PFlowCkfTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
        pixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits",
        stripHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits",
        eventOfHits = "hltIter0PFlowCkfTrackCandidatesMkFitEventOfHits",
        seeds = "hltIter0PFlowCkfTrackCandidatesMkFitSeeds",
        config = ('', 'hltIter0PFlowTrackCandidatesMkFitConfig'),
        minGoodStripCharge = dict(refToPSet_ = 'HLTSiStripClusterChargeCutNone'),
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

    replaceWith = (process.hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits +
                   process.hltSiStripRecHits +
                   process.hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits +
                   process.hltIter0PFlowCkfTrackCandidatesMkFitEventOfHits +
                   process.hltIter0PFlowCkfTrackCandidatesMkFitSeeds +
                   process.hltIter0PFlowCkfTrackCandidatesMkFit +
                   process.hltIter0PFlowCkfTrackCandidates)

    process.HLTIterativeTrackingIteration0.replace(process.hltIter0PFlowCkfTrackCandidates, replaceWith)

    for path in process.paths_().values():
      if not path.contains(process.HLTIterativeTrackingIteration0) and path.contains(process.hltIter0PFlowCkfTrackCandidates):
        path.replace(process.hltIter0PFlowCkfTrackCandidates, replaceWith)

    process.hltIter0PFlowTrackCutClassifier.mva.maxChi2 = cms.vdouble( 999.0, 999.0, 99.0 )
    process.hltIter0PFlowTrackCutClassifier.mva.maxChi2n = cms.vdouble( 999.0, 999.0, 999.0 )
    process.hltIter0PFlowTrackCutClassifier.mva.dr_par = cms.PSet( 
        d0err = cms.vdouble( 0.003, 0.003, 0.003 ),
        dr_par1 = cms.vdouble( 3.40282346639E38, 0.6, 0.6 ),
        dr_par2 = cms.vdouble( 3.40282346639E38, 0.45, 0.45 ),
        dr_exp = cms.vint32( 4, 4, 4 ),
        d0err_par = cms.vdouble( 0.001, 0.001, 0.001 )
    )
    process.hltIter0PFlowTrackCutClassifier.mva.dz_par = cms.PSet( 
        dz_par1 = cms.vdouble( 3.40282346639E38, 0.6, 0.6 ),
        dz_par2 = cms.vdouble( 3.40282346639E38, 0.51, 0.51 ),
        dz_exp = cms.vint32( 4, 4, 4 )
    )

    if hasattr(process, 'HLTIterativeTrackingIteration0SerialSync'):
        process.hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHitsSerialSync = process.hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits.clone(
            hits = "hltSiPixelRecHitsSerialSync",
            clusters = "hltSiPixelClustersSerialSync",
        )
        process.hltIter0PFlowCkfTrackCandidatesMkFitEventOfHitsSerialSync = process.hltIter0PFlowCkfTrackCandidatesMkFitEventOfHits.clone(
            pixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHitsSerialSync",
        )
        process.hltIter0PFlowCkfTrackCandidatesMkFitSeedsSerialSync = process.hltIter0PFlowCkfTrackCandidatesMkFitSeeds.clone(
            seeds = "hltIter0PFLowPixelSeedsFromPixelTracksSerialSync",
        )
        process.hltIter0PFlowCkfTrackCandidatesMkFitSerialSync = process.hltIter0PFlowCkfTrackCandidatesMkFit.clone(
            pixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHitsSerialSync",
            eventOfHits = "hltIter0PFlowCkfTrackCandidatesMkFitEventOfHitsSerialSync",
            seeds = "hltIter0PFlowCkfTrackCandidatesMkFitSeedsSerialSync",
        )
        process.hltIter0PFlowCkfTrackCandidatesSerialSync = process.hltIter0PFlowCkfTrackCandidates.clone(
            seeds = "hltIter0PFLowPixelSeedsFromPixelTracksSerialSync",
            mkFitEventOfHits = "hltIter0PFlowCkfTrackCandidatesMkFitEventOfHitsSerialSync",
            mkFitPixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHitsSerialSync",
            mkFitSeeds = "hltIter0PFlowCkfTrackCandidatesMkFitSeedsSerialSync",
            tracks = "hltIter0PFlowCkfTrackCandidatesMkFitSerialSync",
        )

        process.hltIter0PFlowTrackCutClassifierSerialSync.mva.maxChi2 = cms.vdouble( 999.0, 999.0, 99.0 )
        process.hltIter0PFlowTrackCutClassifierSerialSync.mva.maxChi2n = cms.vdouble( 999.0, 999.0, 999.0 )
        process.hltIter0PFlowTrackCutClassifierSerialSync.mva.dr_par = cms.PSet( 
            d0err = cms.vdouble( 0.003, 0.003, 0.003 ),
            dr_par1 = cms.vdouble( 3.40282346639E38, 0.6, 0.6 ),
            dr_par2 = cms.vdouble( 3.40282346639E38, 0.45, 0.45 ),
            dr_exp = cms.vint32( 4, 4, 4 ),
            d0err_par = cms.vdouble( 0.001, 0.001, 0.001 )
        )
        process.hltIter0PFlowTrackCutClassifierSerialSync.mva.dz_par = cms.PSet( 
            dz_par1 = cms.vdouble( 3.40282346639E38, 0.6, 0.6 ),
            dz_par2 = cms.vdouble( 3.40282346639E38, 0.51, 0.51 ),
            dz_exp = cms.vint32( 4, 4, 4 )
        )
        process.HLTDoLocalStripSequenceSerialSync += process.hltSiStripRecHits

        replaceWithSerialSync = (process.hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHitsSerialSync +
                                 process.hltSiStripRecHits +
                                 process.hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits +
                                 process.hltIter0PFlowCkfTrackCandidatesMkFitEventOfHitsSerialSync +
                                 process.hltIter0PFlowCkfTrackCandidatesMkFitSeedsSerialSync +
                                 process.hltIter0PFlowCkfTrackCandidatesMkFitSerialSync +
                                 process.hltIter0PFlowCkfTrackCandidatesSerialSync)

        process.HLTIterativeTrackingIteration0SerialSync.replace(process.hltIter0PFlowCkfTrackCandidatesSerialSync, replaceWithSerialSync)

        for path in process.paths_().values():
            if not path.contains(process.HLTIterativeTrackingIteration0SerialSync) and path.contains(process.hltIter0PFlowCkfTrackCandidatesSerialSync):
                path.replace(process.hltIter0PFlowCkfTrackCandidatesSerialSync, replaceWithSerialSync)

    return process

def customizeHLTDoubletRecoveryToMkFit(process):

    # if any of the following objects does not exist, do not apply any customisation
    for objLabel in [
        'HLTIterativeTrackingDoubletRecovery',
        'hltDoubletRecoveryPFlowCkfTrackCandidates',
    ]:
        if not hasattr(process, objLabel):
            print(f'# WARNING: customizeHLTDoubletRecoveryToMkFit failed (object with label "{objLabel}" not found) - no customisation applied !')
            return process

    process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSeeds = mkFitSeedConverter_cfi.mkFitSeedConverter.clone(
        seeds = "hltDoubletRecoveryPFlowPixelSeeds",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
    )
    process.hltDoubletRecoveryPFlowTrackCandidatesMkFitConfig = mkFitIterationConfigESProducer_cfi.mkFitIterationConfigESProducer.clone(
        ComponentName = 'hltDoubletRecoveryPFlowTrackCandidatesMkFitConfig',
        config = 'RecoTracker/MkFit/data/mkfit-phase1-hltdr.json',
        minPt = 0.7,
    )
    process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
        pixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits",
        stripHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits",
        eventOfHits = "hltIter0PFlowCkfTrackCandidatesMkFitEventOfHits",
        seeds = "hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSeeds",
        config = cms.ESInputTag('', 'hltDoubletRecoveryPFlowTrackCandidatesMkFitConfig'),
        minGoodStripCharge = dict(refToPSet_ = 'HLTSiStripClusterChargeCutNone'),
    )
    process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFit.clustersToSkip = "hltDoubletRecoveryClustersRefRemoval"
    process.hltDoubletRecoveryPFlowCkfTrackCandidates = mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
        seeds = "hltDoubletRecoveryPFlowPixelSeeds",
        mkFitEventOfHits = "hltIter0PFlowCkfTrackCandidatesMkFitEventOfHits",
        mkFitPixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits",
        mkFitStripHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits",
        mkFitSeeds = "hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSeeds",
        tracks = "hltDoubletRecoveryPFlowCkfTrackCandidatesMkFit",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
        propagatorAlong = ":PropagatorWithMaterialParabolicMf",
        propagatorOpposite = ":PropagatorWithMaterialParabolicMfOpposite",
    )
    replaceWith = (process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSeeds +
                   process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFit +
                   process.hltDoubletRecoveryPFlowCkfTrackCandidates)
    process.HLTIterativeTrackingDoubletRecovery.replace(process.hltDoubletRecoveryPFlowCkfTrackCandidates, replaceWith)
    for path in process.paths_().values():
      if not path.contains(process.HLTIterativeTrackingDoubletRecovery) and path.contains(process.hltDoubletRecoveryPFlowCkfTrackCandidates):
        path.replace(process.hltDoubletRecoveryPFlowCkfTrackCandidates, replaceWith)

    if hasattr(process, 'HLTIterativeTrackingDoubletRecoverySerialSync'):
        process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSeedsSerialSync = process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSeeds.clone(
            seeds = "hltDoubletRecoveryPFlowPixelSeedsSerialSync"
        )
        process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSerialSync = process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFit.clone(
            pixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHitsSerialSync",
            stripHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits",
            eventOfHits = "hltIter0PFlowCkfTrackCandidatesMkFitEventOfHitsSerialSync",
            seeds = "hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSeedsSerialSync",
            config = cms.ESInputTag('', 'hltDoubletRecoveryPFlowTrackCandidatesMkFitConfig'),
            minGoodStripCharge = dict(refToPSet_ = 'HLTSiStripClusterChargeCutNone'),        
        )
        process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSerialSync.clustersToSkip = "hltDoubletRecoveryClustersRefRemovalSerialSync"
        process.hltDoubletRecoveryPFlowCkfTrackCandidatesSerialSync = process.hltDoubletRecoveryPFlowCkfTrackCandidates.clone(
            seeds = "hltDoubletRecoveryPFlowPixelSeedsSerialSync",
            mkFitEventOfHits = "hltIter0PFlowCkfTrackCandidatesMkFitEventOfHitsSerialSync",
            mkFitPixelHits = "hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHitsSerialSync",
            mkFitSeeds = "hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSeedsSerialSync",
            tracks = "hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSerialSync",
        )
        replaceWithSerialSync = (process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSeedsSerialSync +
                                 process.hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSerialSync +
                                 process.hltDoubletRecoveryPFlowCkfTrackCandidatesSerialSync)
        process.HLTIterativeTrackingDoubletRecoverySerialSync.replace(process.hltDoubletRecoveryPFlowCkfTrackCandidatesSerialSync, replaceWithSerialSync)
        for path in process.paths_().values():
            if not path.contains(process.HLTIterativeTrackingDoubletRecoverySerialSync) and path.contains(process.hltDoubletRecoveryPFlowCkfTrackCandidatesSerialSync):
                path.replace(process.hltDoubletRecoveryPFlowCkfTrackCandidatesSerialSync, replaceWithSerialSync)

    process.hltDoubletRecoveryPFlowTrackCutClassifier.mva.maxChi2 = cms.vdouble( 999.0, 999.0, 4.9 )
    process.hltDoubletRecoveryPFlowTrackCutClassifier.mva.maxChi2n = cms.vdouble( 999.0, 999.0, 0.7 )
    process.hltDoubletRecoveryPFlowTrackCutClassifier.mva.dr_par = cms.PSet( 
        d0err = cms.vdouble( 0.003, 0.003, 0.003 ),
        dr_par1 = cms.vdouble( 3.40282346639E38, 0.45, 0.45 ),
        dr_par2 = cms.vdouble( 3.40282346639E38, 0.34, 0.34 ),
        dr_exp = cms.vint32( 4, 4, 4 ),
        d0err_par = cms.vdouble( 0.001, 0.001, 0.001 )
    )
    process.hltDoubletRecoveryPFlowTrackCutClassifier.mva.dz_par = cms.PSet( 
        dz_par1 = cms.vdouble( 3.40282346639E38, 0.45, 0.45 ),
        dz_par2 = cms.vdouble( 3.40282346639E38, 0.39, 0.39 ),
        dz_exp = cms.vint32( 4, 4, 4 )
    )
    process.hltDoubletRecoveryPFlowTrackCutClassifier.mva.min3DLayers = cms.vint32( 0, 0, 3 )
    process.hltDoubletRecoveryPFlowTrackCutClassifier.mva.minLayers = cms.vint32( 0, 0, 4 )
    process.hltDoubletRecoveryPFlowTrackCutClassifier.mva.minHits = cms.vint32( 0, 0, 5 )
    process.hltDoubletRecoveryPFlowTrackCutClassifier.mva.maxLostLayers = cms.vint32( 0, 0, 0 )

    if hasattr(process, 'hltDoubletRecoveryPFlowTrackCutClassifierSerialSync'):
        process.hltDoubletRecoveryPFlowTrackCutClassifierSerialSync.mva.maxChi2 = cms.vdouble( 999.0, 99.0, 4.9 )
        process.hltDoubletRecoveryPFlowTrackCutClassifierSerialSync.mva.maxChi2n = cms.vdouble( 999.0, 999.0, 0.7 )
        process.hltDoubletRecoveryPFlowTrackCutClassifierSerialSync.mva.dr_par = cms.PSet( 
            d0err = cms.vdouble( 0.003, 0.003, 0.003 ),
            d0err_par = cms.vdouble( 0.001, 0.001, 0.001 ),
            dr_par1 = cms.vdouble( 3.40282346639E38, 0.45, 0.45 ),
            dr_par2 = cms.vdouble( 3.40282346639E38, 0.34, 0.34 ),
            dr_exp = cms.vint32( 4, 4, 4 ),
        )
        process.hltDoubletRecoveryPFlowTrackCutClassifierSerialSync.mva.dz_par = cms.PSet( 
            dz_par1 = cms.vdouble( 3.40282346639E38, 0.45, 0.45 ),
            dz_par2 = cms.vdouble( 3.40282346639E38, 0.39, 0.39 ),
            dz_exp = cms.vint32( 4, 4, 4 )
        )
        process.hltDoubletRecoveryPFlowTrackCutClassifierSerialSync.mva.min3DLayers = cms.vint32( 0, 0, 3 )
        process.hltDoubletRecoveryPFlowTrackCutClassifierSerialSync.mva.minLayers = cms.vint32( 0, 0, 4 )
        process.hltDoubletRecoveryPFlowTrackCutClassifierSerialSync.mva.minHits = cms.vint32( 0, 0, 5 )
        process.hltDoubletRecoveryPFlowTrackCutClassifierSerialSync.mva.maxLostLayers = cms.vint32( 0, 0, 0 )

    return process

def modifyMinOutputModuleForTrackingValidation(process, filename="output.root"):

    for objLabel in [
        'hltOutputMinimal',
    ]:
        if not hasattr(process, objLabel):
            print(f'# WARNING: customize command failed (object with label "{objLabel}" not found) - no customisation applied !')
            return process

    process.load('Configuration.EventContent.EventContent_cff')
    process.hltOutputMinimal.outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands
    process.hltOutputMinimal.fileName = filename
    process.schedule.remove( process.DQMOutput )
    return process
