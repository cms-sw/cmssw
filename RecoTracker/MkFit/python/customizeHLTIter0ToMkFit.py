import FWCore.ParameterSet.Config as cms

import RecoTracker.MkFit.mkFitGeometryESProducer_cfi as mkFitGeometryESProducer_cfi
import RecoTracker.MkFit.mkFitSiPixelHitConverter_cfi as mkFitSiPixelHitConverter_cfi
import RecoTracker.MkFit.mkFitSiStripHitConverter_cfi as mkFitSiStripHitConverter_cfi
import RecoTracker.MkFit.mkFitSiStripHitConverterFromClusters_cfi as mkFitSiStripHitConverterFromClusters_cfi
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
    process.hltSiStripRawToClustersFacility.Clusterizer.MaxClusterSize = 8

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
        config = 'RecoTracker/MkFit/data/mkfit-phase1-initialStep.json',
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

    process.HLTDoLocalStripSequence += process.hltSiStripRecHits

    replaceWith = (process.hltIter0PFlowCkfTrackCandidatesMkFitSiPixelHits +
                   process.hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits +
                   process.hltIter0PFlowCkfTrackCandidatesMkFitEventOfHits +
                   process.hltIter0PFlowCkfTrackCandidatesMkFitSeeds +
                   process.hltIter0PFlowCkfTrackCandidatesMkFit +
                   process.hltIter0PFlowCkfTrackCandidates)

    process.HLTIterativeTrackingIteration0.replace(process.hltIter0PFlowCkfTrackCandidates, replaceWith)

    for path in process.paths_().values():
      if not path.contains(process.HLTIterativeTrackingIteration0) and path.contains(process.hltIter0PFlowCkfTrackCandidates):
        path.replace(process.hltIter0PFlowCkfTrackCandidates, replaceWith)

    process.hltIter0PFlowTrackCandidatesMkFitConfig.config = 'RecoTracker/MkFit/data/mkfit-phase1-hltiter0.json'

    process.hltIter0PFlowTrackCutClassifier.mva.maxChi2 = cms.vdouble( 999.0, 25.0, 99.0 )

    process.hltIter0PFlowTrackCutClassifier.mva.maxChi2n = cms.vdouble( 1.2, 1.0, 999.0 )

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

    if hasattr(process, 'hltIter0PFlowTrackCutClassifierSerialSync'):
        process.hltIter0PFlowTrackCutClassifierSerialSync.mva.maxChi2 = cms.vdouble( 999.0, 25.0, 99.0 )
        process.hltIter0PFlowTrackCutClassifierSerialSync.mva.maxChi2n = cms.vdouble( 1.2, 1.0, 999.0 )
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

    return process

def customizeHLTStripHitsFromMkFit(process):

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

    process.hltIter0PFlowCkfTrackCandidatesMkFitSiStripHits = mkFitSiStripHitConverterFromClusters_cfi.mkFitSiStripHitConverterFromClusters.clone(
        clusters = "hltSiStripRawToClustersFacility",
        ttrhBuilder = ":hltESPTTRHBWithTrackAngle",
        StripCPE = process.hltSiStripRecHits.StripCPE,
        minGoodStripCharge = dict(refToPSet_ = 'HLTSiStripClusterChargeCutLoose'),
        doMatching = False,
    )

    delattr(process, "hltSiStripRecHits")

    return process

def customizeHLTSiStripClusterizerOnDemandFalse(process):

    # if any of the following objects does not exist, do not apply any customisation
    for objLabel in [
        'hltSiStripRawToClustersFacility',
    ]:
        if not hasattr(process, objLabel):
            print(f'# WARNING: customize command failed (object with label "{objLabel}" not found) - no customisation applied !')
            return process

    # mkFit needs all clusters, so switch off the on-demand mode
    process.hltSiStripRawToClustersFacility.onDemand = False
    return process

def customizeHLTSiStripClusterizerOnDemandFalseMaxClusterSize8(process):

    for objLabel in [
        'hltSiStripRawToClustersFacility',
    ]:
        if not hasattr(process, objLabel):
            print(f'# WARNING: customize command failed (object with label "{objLabel}" not found) - no customisation applied !')
            return process

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
    process.hltSiStripRawToClustersFacility.Clusterizer.MaxClusterSize = 8

    return process

def modifyMinOutputModuleForTrackingValidation(process, filename="output.root"):

    for objLabel in [
        'hltOutputMinimal',
    ]:
        if not hasattr(process, objLabel):
            print(f'# WARNING: customize command failed (object with label "{objLabel}" not found) - no customisation applied !')
            return process

    process.hltOutputMinimal.outputCommands = cms.untracked.vstring(
        'drop *',
        'keep edmTriggerResults_*_*_*',
        'keep triggerTriggerEvent_*_*_*',
        'keep GlobalAlgBlkBXVector_*_*_*',
        'keep GlobalExtBlkBXVector_*_*_*',
        'keep l1tEGammaBXVector_*_EGamma_*',
        'keep l1tEtSumBXVector_*_EtSum_*',
        'keep l1tJetBXVector_*_Jet_*',
        'keep l1tMuonBXVector_*_Muon_*',
        'keep l1tTauBXVector_*_Tau_*',
        'keep *_*_*_HLTX',
        'drop *_hltHbherecoLegacy_*_*',
        'drop *_hlt*Pixel*SoA*_*_*',
        'keep recoGenParticles_genParticles_*_*',
        'keep TrackingParticles_*_*_*',
        'keep *_*_MergedTrackTruth_*',
        'keep *_simSiPixelDigis_*_*',
        'keep *_simSiStripDigis_*_*',
        'keep PSimHits_g4SimHits_*_*',
        'keep SimTracks_g4SimHits_*_*',
        'keep SimVertexs_g4SimHits_*_*',
        'keep PileupSummaryInfos_addPileupInfo_*_*',
    )
    process.hltOutputMinimal.fileName = filename
    process.schedule.remove( process.DQMOutput )
    return process
