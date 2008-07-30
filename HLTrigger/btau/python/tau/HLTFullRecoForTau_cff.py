import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
trajFilterL25 = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *
trajBuilderL25 = copy.deepcopy(CkfTrajectoryBuilder)
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
trajFilterL3 = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *
trajBuilderL3 = copy.deepcopy(CkfTrajectoryBuilder)
import copy
from RecoTauTag.HLTProducers.TauRegionalPixelSeedGenerator_cfi import *
l25SingleTauPixelSeeds = copy.deepcopy(tauRegionalPixelSeedGenerator)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
ckfTrackCandidatesL25SingleTau = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
ctfWithMaterialTracksL25SingleTau = copy.deepcopy(ctfWithMaterialTracks)
import copy
from RecoTauTag.HLTProducers.TauRegionalPixelSeedGenerator_cfi import *
l3SingleTauPixelSeeds = copy.deepcopy(tauRegionalPixelSeedGenerator)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
ckfTrackCandidatesL3SingleTau = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
ctfWithMaterialTracksL3SingleTau = copy.deepcopy(ctfWithMaterialTracks)
import copy
from RecoTauTag.HLTProducers.TauRegionalPixelSeedGenerator_cfi import *
l25SingleTauMETPixelSeeds = copy.deepcopy(tauRegionalPixelSeedGenerator)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
ckfTrackCandidatesL25SingleTauMET = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
ctfWithMaterialTracksL25SingleTauMET = copy.deepcopy(ctfWithMaterialTracks)
import copy
from RecoTauTag.HLTProducers.TauRegionalPixelSeedGenerator_cfi import *
l3SingleTauMETPixelSeeds = copy.deepcopy(tauRegionalPixelSeedGenerator)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
ckfTrackCandidatesL3SingleTauMET = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
ctfWithMaterialTracksL3SingleTauMET = copy.deepcopy(ctfWithMaterialTracks)
from HLTrigger.Configuration.common.Vertexing_cff import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from HLTrigger.Configuration.common.RecoJetMET_cff import *
import copy
from RecoTauTag.HLTProducers.TauRegionalPixelSeedGenerator_cfi import *
hltTauRegionalPixelSeedL25ElectronTau = copy.deepcopy(tauRegionalPixelSeedGenerator)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
hltCkfTrackCandidatesL25ElectronTau = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
hltCtfWithMaterialTracksL25ElectronTau = copy.deepcopy(ctfWithMaterialTracks)
import copy
from RecoTauTag.HLTProducers.TauRegionalPixelSeedGenerator_cfi import *
hltTauRegionalPixelSeedsL3ElectronTau = copy.deepcopy(tauRegionalPixelSeedGenerator)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
hltCkfTrackCandidatesL3ElectronTau = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
hltCtfWithMaterialTracksL3ElectronTau = copy.deepcopy(ctfWithMaterialTracks)
ckfTracksL25SingleTau = cms.Sequence(l25SingleTauPixelSeeds*ckfTrackCandidatesL25SingleTau*ctfWithMaterialTracksL25SingleTau)
ckfTracksL3SingleTau = cms.Sequence(l3SingleTauPixelSeeds*ckfTrackCandidatesL3SingleTau*ctfWithMaterialTracksL3SingleTau)
ckfTracksL25SingleTauMET = cms.Sequence(l25SingleTauMETPixelSeeds*ckfTrackCandidatesL25SingleTauMET*ctfWithMaterialTracksL25SingleTauMET)
ckfTracksL3SingleTauMET = cms.Sequence(l3SingleTauMETPixelSeeds*ckfTrackCandidatesL3SingleTauMET*ctfWithMaterialTracksL3SingleTauMET)
hltTracksL25ElectronTau = cms.Sequence(hltTauRegionalPixelSeedL25ElectronTau*hltCkfTrackCandidatesL25ElectronTau*hltCtfWithMaterialTracksL25ElectronTau)
hltTracksL3ElectronTau = cms.Sequence(hltTauRegionalPixelSeedsL3ElectronTau*hltCkfTrackCandidatesL3ElectronTau*hltCtfWithMaterialTracksL3ElectronTau)
trajFilterL25.ComponentName = 'trajFilterL25'
trajFilterL25.filterPset.minPt = 5.0
trajFilterL25.filterPset.maxNumberOfHits = 7
trajBuilderL25.ComponentName = 'trajBuilderL25'
trajBuilderL25.trajectoryFilterName = 'trajFilterL25'
trajBuilderL25.maxCand = 1
trajBuilderL25.alwaysUseInvalidHits = False
trajFilterL3.ComponentName = 'trajFilterL3'
trajFilterL3.filterPset.maxNumberOfHits = 7
trajBuilderL3.ComponentName = 'trajBuilderL3'
trajBuilderL3.trajectoryFilterName = 'trajFilterL3'
trajBuilderL3.alwaysUseInvalidHits = False
l25SingleTauPixelSeeds.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("ecalSingleTauIsolated","Isolated")
ckfTrackCandidatesL25SingleTau.SeedProducer = 'l25SingleTauPixelSeeds'
ckfTrackCandidatesL25SingleTau.TrajectoryBuilder = 'trajBuilderL25'
ctfWithMaterialTracksL25SingleTau.src = 'ckfTrackCandidatesL25SingleTau'
l3SingleTauPixelSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
    precise = cms.bool(True),
    deltaPhiRegion = cms.double(0.5),
    originHalfLength = cms.double(0.2),
    originRadius = cms.double(0.2),
    deltaEtaRegion = cms.double(0.5),
    ptMin = cms.double(1.0),
    JetSrc = cms.InputTag("isolatedL25SingleTau"),
    originZPos = cms.double(0.0),
    vertexSrc = cms.InputTag("pixelVertices")
)
ckfTrackCandidatesL3SingleTau.SeedProducer = 'l3SingleTauPixelSeeds'
ckfTrackCandidatesL3SingleTau.TrajectoryBuilder = 'trajBuilderL3'
ctfWithMaterialTracksL3SingleTau.src = 'ckfTrackCandidatesL3SingleTau'
l25SingleTauMETPixelSeeds.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("ecalSingleTauMETIsolated","Isolated")
ckfTrackCandidatesL25SingleTauMET.SeedProducer = 'l25SingleTauMETPixelSeeds'
ckfTrackCandidatesL25SingleTauMET.TrajectoryBuilder = 'trajBuilderL25'
ctfWithMaterialTracksL25SingleTauMET.src = 'ckfTrackCandidatesL25SingleTauMET'
l3SingleTauMETPixelSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
    precise = cms.bool(True),
    deltaPhiRegion = cms.double(0.5),
    originHalfLength = cms.double(0.2),
    originRadius = cms.double(0.2),
    deltaEtaRegion = cms.double(0.5),
    ptMin = cms.double(1.0),
    JetSrc = cms.InputTag("isolatedL25SingleTauMET"),
    originZPos = cms.double(0.0),
    vertexSrc = cms.InputTag("pixelVertices")
)
ckfTrackCandidatesL3SingleTauMET.SeedProducer = 'l3SingleTauMETPixelSeeds'
ckfTrackCandidatesL3SingleTauMET.TrajectoryBuilder = 'trajBuilderL3'
ctfWithMaterialTracksL3SingleTauMET.src = 'ckfTrackCandidatesL3SingleTauMET'
hltTauRegionalPixelSeedL25ElectronTau.RegionFactoryPSet.RegionPSet.ptMin = 5.0
hltTauRegionalPixelSeedL25ElectronTau.RegionFactoryPSet.RegionPSet.JetSrc = 'hltL2TauJetsProviderElectronTau'
hltCkfTrackCandidatesL25ElectronTau.SeedProducer = 'hltTauRegionalPixelSeedL25ElectronTau'
hltCkfTrackCandidatesL25ElectronTau.TrajectoryBuilder = 'trajBuilderL25'
hltCtfWithMaterialTracksL25ElectronTau.src = 'hltCkfTrackCandidatesL25ElectronTau'
hltTauRegionalPixelSeedsL3ElectronTau.RegionFactoryPSet.RegionPSet.JetSrc = 'hltIsolatedTauJetsSelectorL25ElectronTau'
hltTauRegionalPixelSeedsL3ElectronTau.RegionFactoryPSet.RegionPSet.ptMin = 1.0
hltTauRegionalPixelSeedsL3ElectronTau.RegionFactoryPSet.RegionPSet.deltaEtaRegion = 0.5
hltTauRegionalPixelSeedsL3ElectronTau.RegionFactoryPSet.RegionPSet.deltaPhiRegion = 0.5
hltCkfTrackCandidatesL3ElectronTau.SeedProducer = 'hltTauRegionalPixelSeedsL3ElectronTau'
hltCkfTrackCandidatesL3ElectronTau.TrajectoryBuilder = 'trajBuilderL3'
hltCtfWithMaterialTracksL3ElectronTau.src = 'hltCkfTrackCandidatesL3ElectronTau'

