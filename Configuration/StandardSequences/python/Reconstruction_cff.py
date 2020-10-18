import FWCore.ParameterSet.Config as cms

from RecoLuminosity.LumiProducer.lumiProducer_cff import *
from RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi import *
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import *
from RecoLocalFastTime.Configuration.RecoLocalFastTime_cff import *
from RecoMTD.Configuration.RecoMTD_cff import *
from RecoTracker.Configuration.RecoTracker_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *
from TrackingTools.Configuration.TrackingTools_cff import *
from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
from RecoHGCal.Configuration.recoHGCAL_cff import *

from Configuration.Eras.Modifier_fastSim_cff import fastSim

siPixelClusterShapeCachePreSplitting = siPixelClusterShapeCache.clone(
    src = 'siPixelClustersPreSplitting'
    )

# Global  reco
from RecoEcal.Configuration.RecoEcal_cff import *
from RecoJets.Configuration.CaloTowersRec_cff import *
from RecoMET.Configuration.RecoMET_cff import *
from RecoMuon.Configuration.RecoMuon_cff import *
# Higher level objects
from RecoVertex.Configuration.RecoVertex_cff import *
from RecoEgamma.Configuration.RecoEgamma_cff import *
from RecoPixelVertexing.Configuration.RecoPixelVertexing_cff import *


from RecoJets.Configuration.RecoJetsGlobal_cff import *
from RecoMET.Configuration.RecoPFMET_cff import *
from RecoBTag.Configuration.RecoBTag_cff import *
#
# please understand that division global,highlevel is completely fake !
#
#local reconstruction
from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
from RecoParticleFlow.Configuration.RecoParticleFlow_cff import *
from RecoPPS.Configuration.recoCTPPS_cff import *
#
# new tau configuration
#
from RecoTauTag.Configuration.RecoPFTauTag_cff import *
# Also BeamSpot
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *

from RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi import *

# Low pT electrons
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronSequence_cff import *

# Conversions from lowPtGsfTracks
from RecoEgamma.EgammaPhotonProducers.conversionOpenTrackSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.gsfTracksOpenConversionSequence_cff import *


localrecoTask = cms.Task(bunchSpacingProducer,trackerlocalrecoTask,muonlocalrecoTask,calolocalrecoTask,castorreco)
localreco = cms.Sequence(localrecoTask)
localreco_HcalNZSTask = cms.Task(bunchSpacingProducer,trackerlocalrecoTask,muonlocalrecoTask,calolocalrecoTaskNZS,castorreco)
localreco_HcalNZS = cms.Sequence(localreco_HcalNZSTask)

_run3_localrecoTask = localrecoTask.copyAndExclude([castorreco])
_run3_localreco_HcalNZSTask = localreco_HcalNZSTask.copyAndExclude([castorreco])
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toReplaceWith(localrecoTask, _run3_localrecoTask)
run3_common.toReplaceWith(localreco_HcalNZSTask, _run3_localreco_HcalNZSTask)

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
_phase2_timing_layer_localrecoTask = _run3_localrecoTask.copy()
_phase2_timing_layer_localrecoTask.add(fastTimingLocalRecoTask)
_phase2_timing_layer_localreco_HcalNZSTask = _run3_localreco_HcalNZSTask.copy()
_phase2_timing_layer_localreco_HcalNZSTask.add(fastTimingLocalRecoTask)
phase2_timing_layer.toReplaceWith(localrecoTask,_phase2_timing_layer_localrecoTask)
phase2_timing_layer.toReplaceWith(localreco_HcalNZSTask,_phase2_timing_layer_localreco_HcalNZSTask)

_ctpps_2016_localrecoTask = localrecoTask.copy()
_ctpps_2016_localrecoTask.add(recoCTPPSTask)
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
ctpps_2016.toReplaceWith(localrecoTask, _ctpps_2016_localrecoTask)

_ctpps_2016_localreco_HcalNZSTask = localreco_HcalNZSTask.copy()
_ctpps_2016_localreco_HcalNZSTask.add(recoCTPPSTask)
ctpps_2016.toReplaceWith(localreco_HcalNZSTask, _ctpps_2016_localreco_HcalNZSTask)

###########################################
# no castor, zdc, Totem/CTPPS RP in FastSim
###########################################
_fastSim_localrecoTask = localrecoTask.copyAndExclude([
    castorreco,
    totemRPLocalReconstructionTask,totemTimingLocalReconstructionTask,ctppsDiamondLocalReconstructionTask,
    ctppsLocalTrackLiteProducer,ctppsPixelLocalReconstructionTask,ctppsProtons,
    trackerlocalrecoTask
])
fastSim.toReplaceWith(localrecoTask, _fastSim_localrecoTask)

#
# temporarily switching off recoGenJets; since this are MC and wil be moved to a proper sequence
#

from RecoLocalCalo.Castor.Castor_cff import *
from RecoLocalCalo.Configuration.hcalGlobalReco_cff import *

globalreco_trackingTask = cms.Task(offlineBeamSpotTask,
                          MeasurementTrackerEventPreSplitting, # unclear where to put this
                          siPixelClusterShapeCachePreSplitting, # unclear where to put this
                          standalonemuontrackingTask,
                          trackingGlobalRecoTask,
                          hcalGlobalRecoTask,
                          vertexrecoTask)
_globalreco_tracking_LowPUTask = globalreco_trackingTask.copy()
_globalreco_tracking_LowPUTask.replace(trackingGlobalRecoTask, cms.Task(recopixelvertexingTask,trackingGlobalRecoTask))
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(globalreco_trackingTask, _globalreco_tracking_LowPUTask)
##########################################
# offlineBeamSpot is reconstructed before mixing in fastSim
##########################################
_fastSim_globalreco_trackingTask = globalreco_trackingTask.copyAndExclude([offlineBeamSpotTask,MeasurementTrackerEventPreSplitting,siPixelClusterShapeCachePreSplitting])
fastSim.toReplaceWith(globalreco_trackingTask,_fastSim_globalreco_trackingTask)

_phase2_timing_layer_globalreco_trackingTask = globalreco_trackingTask.copy()
_phase2_timing_layer_globalreco_trackingTask.add(fastTimingGlobalRecoTask)
phase2_timing_layer.toReplaceWith(globalreco_trackingTask,_phase2_timing_layer_globalreco_trackingTask)

globalrecoTask = cms.Task(globalreco_trackingTask,
                          particleFlowClusterTask,
                          ecalClustersTask,
                          caloTowersRecTask,
                          egammaGlobalRecoTask,
                          jetGlobalRecoTask,
                          muonGlobalRecoTask,
                          pfTrackingGlobalRecoTask,
                          muoncosmicrecoTask,
                          CastorFullRecoTask)
globalreco = cms.Sequence(globalrecoTask)

_run3_globalrecoTask = globalrecoTask.copyAndExclude([CastorFullRecoTask])
run3_common.toReplaceWith(globalrecoTask, _run3_globalrecoTask)

_fastSim_globalrecoTask = globalrecoTask.copyAndExclude([CastorFullRecoTask,muoncosmicrecoTask])
# insert the few tracking modules to be run after mixing back in the globalreco sequence
_fastSim_globalrecoTask.add(newCombinedSeeds,trackExtrapolator,caloTowerForTrk,firstStepPrimaryVerticesUnsorted,ak4CaloJetsForTrk,initialStepTrackRefsForJets,firstStepPrimaryVertices)
fastSim.toReplaceWith(globalrecoTask,_fastSim_globalrecoTask)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
_phase2HGALRecoTask = globalrecoTask.copy()
_phase2HGALRecoTask.add(iterTICLTask)
phase2_hgcal.toReplaceWith(globalrecoTask, _phase2HGALRecoTask)

from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
_phase2HFNoseRecoTask = globalrecoTask.copy()
_phase2HFNoseRecoTask.add(iterHFNoseTICLTask)
phase2_hfnose.toReplaceWith(globalrecoTask, _phase2HFNoseRecoTask)


globalreco_plusPLTask= cms.Task(globalrecoTask,ctfTracksPixelLessTask)
globalreco_plusPL= cms.Sequence(globalreco_plusPLTask)

reducedRecHitsTask = cms.Task( reducedEcalRecHitsTask , reducedHcalRecHitsTask )
reducedRecHits = cms.Sequence (reducedRecHitsTask)

highlevelrecoTask = cms.Task(egammaHighLevelRecoPrePFTask,
                             particleFlowRecoTask,
                             egammaHighLevelRecoPostPFTask,
                             muoncosmichighlevelrecoTask,
                             muonshighlevelrecoTask,
                             particleFlowLinksTask,
                             jetHighLevelRecoTask,
                             metrecoPlusHCALNoiseTask,
                             btaggingTask,
                             recoPFMETTask,
                             PFTauTask,
                             reducedRecHitsTask,
                             lowPtGsfElectronTask,
                             conversionOpenTrackTask,
                             gsfTracksOpenConversions
                             )
highlevelreco = cms.Sequence(highlevelrecoTask)

# AA data with pp reco
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from RecoHI.HiTracking.HILowPtConformalPixelTracks_cfi import *
from RecoHI.HiCentralityAlgos.HiCentrality_cfi import hiCentrality
from RecoHI.HiCentralityAlgos.HiClusterCompatibility_cfi import hiClusterCompatibility
_highlevelreco_HITask = highlevelrecoTask.copy()
_highlevelreco_HITask.add(hiConformalPixelTracksTaskPhase1)
_highlevelreco_HITask.add(hiCentrality)
_highlevelreco_HITask.add(hiClusterCompatibility)
(pp_on_XeXe_2017 | pp_on_AA_2018).toReplaceWith(highlevelrecoTask, _highlevelreco_HITask)
pp_on_AA_2018.toReplaceWith(highlevelrecoTask,highlevelrecoTask.copyAndExclude([PFTauTask]))

# not commisoned and not relevant in FastSim (?):
_fastSim_highlevelrecoTask = highlevelrecoTask.copyAndExclude([muoncosmichighlevelrecoTask])
fastSim.toReplaceWith(highlevelrecoTask,_fastSim_highlevelrecoTask)


from FWCore.Modules.logErrorHarvester_cfi import *

# "Export" Section
reconstructionTask     = cms.Task(localrecoTask,globalrecoTask,highlevelrecoTask,logErrorHarvester)
reconstruction         = cms.Sequence(reconstructionTask)

#logErrorHarvester should only wait for items produced in the reconstruction sequence
_modulesInReconstruction = list()
reconstructionTask.visit(cms.ModuleNamesFromGlobalsVisitor(globals(),_modulesInReconstruction))
logErrorHarvester.includeModules = cms.untracked.vstring(set(_modulesInReconstruction))

reconstruction_trackingOnlyTask = cms.Task(localrecoTask,globalreco_trackingTask)
#calo parts removed as long as tracking is not running jetCore in phase2
trackingPhase2PU140.toReplaceWith(reconstruction_trackingOnlyTask,
                                  reconstruction_trackingOnlyTask.copyAndExclude([hgcalLocalRecoTask,castorreco]))
reconstruction_trackingOnly = cms.Sequence(reconstruction_trackingOnlyTask)
reconstruction_pixelTrackingOnlyTask = cms.Task(
    pixeltrackerlocalrecoTask,
    offlineBeamSpotTask,
    siPixelClusterShapeCachePreSplitting,
    recopixelvertexingTask
)
reconstruction_pixelTrackingOnly = cms.Sequence(reconstruction_pixelTrackingOnlyTask)

reconstruction_ecalOnlyTask = cms.Task(
    bunchSpacingProducer,
    offlineBeamSpot,
    ecalOnlyLocalRecoTask,
    pfClusteringPSTask,
    pfClusteringECALTask,
    particleFlowSuperClusterECALOnly
)
reconstruction_ecalOnly = cms.Sequence(reconstruction_ecalOnlyTask)

reconstruction_hcalOnlyTask = cms.Task(
    bunchSpacingProducer,
    offlineBeamSpot,
    hcalOnlyLocalRecoTask,
    pfClusteringHBHEHFOnlyTask
)

reconstruction_hcalOnly = cms.Sequence(reconstruction_hcalOnlyTask)

#need a fully expanded sequence copy
modulesToRemove = list() # copy does not work well
noTrackingAndDependent = list()
noTrackingAndDependent.append(siPixelClustersPreSplitting)
noTrackingAndDependent.append(siStripZeroSuppression)
noTrackingAndDependent.append(siStripClusters)
noTrackingAndDependent.append(initialStepSeedLayersPreSplitting)
noTrackingAndDependent.append(trackerClusterCheckPreSplitting)
noTrackingAndDependent.append(initialStepTrackingRegionsPreSplitting)
noTrackingAndDependent.append(initialStepHitDoubletsPreSplitting)
noTrackingAndDependent.append(initialStepHitTripletsPreSplitting)
noTrackingAndDependent.append(initialStepSeedsPreSplitting)
noTrackingAndDependent.append(initialStepTrackCandidatesPreSplitting)
noTrackingAndDependent.append(initialStepTracksPreSplitting)
noTrackingAndDependent.append(firstStepPrimaryVerticesPreSplitting)
noTrackingAndDependent.append(initialStepTrackRefsForJetsPreSplitting)
noTrackingAndDependent.append(caloTowerForTrkPreSplitting)
noTrackingAndDependent.append(ak4CaloJetsForTrkPreSplitting)
noTrackingAndDependent.append(jetsForCoreTrackingPreSplitting)
noTrackingAndDependent.append(siPixelClusterShapeCachePreSplitting)
noTrackingAndDependent.append(siPixelClusters)
noTrackingAndDependent.append(clusterSummaryProducer)
noTrackingAndDependent.append(siPixelRecHitsPreSplitting)
noTrackingAndDependent.append(MeasurementTrackerEventPreSplitting)
noTrackingAndDependent.append(PixelLayerTriplets)
noTrackingAndDependent.append(pixelTracks)
noTrackingAndDependent.append(pixelVertices)
modulesToRemove.append(dt1DRecHits)
modulesToRemove.append(dt1DCosmicRecHits)
modulesToRemove.append(csc2DRecHits)
modulesToRemove.append(rpcRecHits)
modulesToRemove.append(gemRecHits)
#modulesToRemove.append(ecalGlobalUncalibRecHit)
modulesToRemove.append(ecalMultiFitUncalibRecHit)
modulesToRemove.append(ecalDetIdToBeRecovered)
modulesToRemove.append(ecalRecHit)
modulesToRemove.append(ecalCompactTrigPrim)
modulesToRemove.append(ecalTPSkim)
modulesToRemove.append(ecalPreshowerRecHit)
modulesToRemove.append(selectDigi)
modulesToRemove.append(hbheprereco)
modulesToRemove.append(hbhereco)
modulesToRemove.append(hfreco)
modulesToRemove.append(horeco)
modulesToRemove.append(hcalnoise)
modulesToRemove.append(zdcreco)
modulesToRemove.append(castorreco)
##it's OK according to Ronny modulesToRemove.append(CSCHaloData)#needs digis
reconstruction_fromRECO = reconstruction.copyAndExclude(modulesToRemove+noTrackingAndDependent)


#sequences with additional stuff
reconstruction_withPixellessTkTask  = cms.Task(localrecoTask,globalreco_plusPLTask,highlevelrecoTask,logErrorHarvester)
reconstruction_withPixellessTk  = cms.Sequence(reconstruction_withPixellessTkTask)
reconstruction_HcalNZSTask = cms.Task(localreco_HcalNZSTask,globalrecoTask,highlevelrecoTask,logErrorHarvester)
reconstruction_HcalNZS = cms.Sequence(reconstruction_HcalNZSTask)

#sequences without some stuffs
#
reconstruction_woCosmicMuonsTask = cms.Task(localrecoTask,globalrecoTask,highlevelrecoTask,logErrorHarvester)
reconstruction_woCosmicMuons = cms.Sequence(reconstruction_woCosmicMuonsTask)
