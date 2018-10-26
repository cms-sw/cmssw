import FWCore.ParameterSet.Config as cms

from RecoLuminosity.LumiProducer.lumiProducer_cff import *
from RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi import *
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import *
from RecoLocalFastTime.Configuration.RecoLocalFastTime_cff import *
from RecoTracker.Configuration.RecoTracker_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *
from TrackingTools.Configuration.TrackingTools_cff import *
from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *

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
from RecoCTPPS.Configuration.recoCTPPS_cff import *
#
# new tau configuration
#
from RecoTauTag.Configuration.RecoPFTauTag_cff import *
# Also BeamSpot
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *

from RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi import *

# Cosmic During Collisions
from RecoTracker.SpecialSeedGenerators.cosmicDC_cff import *

localreco = cms.Sequence(bunchSpacingProducer+trackerlocalreco+muonlocalreco+calolocalreco+castorreco)
localreco_HcalNZS = cms.Sequence(bunchSpacingProducer+trackerlocalreco+muonlocalreco+calolocalrecoNZS+castorreco)

_run3_localreco = localreco.copyAndExclude([castorreco])
_run3_localreco_HcalNZS = localreco_HcalNZS.copyAndExclude([castorreco])
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toReplaceWith(localreco, _run3_localreco)
run3_common.toReplaceWith(localreco_HcalNZS, _run3_localreco_HcalNZS)

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
_phase2_timing_layer_localreco = _run3_localreco.copy()
_phase2_timing_layer_localreco += fastTimingLocalReco
_phase2_timing_layer_localreco_HcalNZS = _run3_localreco_HcalNZS.copy()
_phase2_timing_layer_localreco_HcalNZS += fastTimingLocalReco
phase2_timing_layer.toReplaceWith(localreco,_phase2_timing_layer_localreco)
phase2_timing_layer.toReplaceWith(localreco_HcalNZS,_phase2_timing_layer_localreco_HcalNZS)

_ctpps_2016_localreco = localreco.copy()
_ctpps_2016_localreco += recoCTPPS
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
ctpps_2016.toReplaceWith(localreco, _ctpps_2016_localreco)

_ctpps_2016_localreco_HcalNZS = localreco_HcalNZS.copy()
_ctpps_2016_localreco_HcalNZS += recoCTPPS
ctpps_2016.toReplaceWith(localreco_HcalNZS, _ctpps_2016_localreco_HcalNZS)

###########################################
# no castor, zdc, Totem/CTPPS RP in FastSim
###########################################
_fastSim_localreco = localreco.copyAndExclude([
    castorreco,
    totemRPLocalReconstruction,totemTimingLocalReconstruction,ctppsDiamondLocalReconstruction,ctppsLocalTrackLiteProducer,ctppsPixelLocalReconstruction,
    trackerlocalreco
])
fastSim.toReplaceWith(localreco, _fastSim_localreco)

#
# temporarily switching off recoGenJets; since this are MC and wil be moved to a proper sequence
#

from RecoLocalCalo.Castor.Castor_cff import *
from RecoLocalCalo.Configuration.hcalGlobalReco_cff import *

globalreco_tracking = cms.Sequence(offlineBeamSpot*
                          MeasurementTrackerEventPreSplitting* # unclear where to put this
                          siPixelClusterShapeCachePreSplitting* # unclear where to put this
                          standalonemuontracking*
                          trackingGlobalReco*
                          hcalGlobalRecoSequence*
                          vertexreco)
_globalreco_tracking_LowPU = globalreco_tracking.copy()
_globalreco_tracking_LowPU.replace(trackingGlobalReco, recopixelvertexing+trackingGlobalReco)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(globalreco_tracking, _globalreco_tracking_LowPU)
##########################################
# offlineBeamSpot is reconstructed before mixing in fastSim
##########################################
_fastSim_globalreco_tracking = globalreco_tracking.copyAndExclude([offlineBeamSpot,MeasurementTrackerEventPreSplitting,siPixelClusterShapeCachePreSplitting])
fastSim.toReplaceWith(globalreco_tracking,_fastSim_globalreco_tracking)

globalreco = cms.Sequence(globalreco_tracking*
                          particleFlowCluster*
                          ecalClusters*
                          caloTowersRec*
                          egammaGlobalReco*
                          jetGlobalReco*
                          muonGlobalReco*
                          pfTrackingGlobalReco*
                          muoncosmicreco*
                          CastorFullReco)

_run3_globalreco = globalreco.copyAndExclude([CastorFullReco])
run3_common.toReplaceWith(globalreco, _run3_globalreco)

_fastSim_globalreco = globalreco.copyAndExclude([CastorFullReco,muoncosmicreco])
# insert the few tracking modules to be run after mixing back in the globalreco sequence
_fastSim_globalreco.insert(0,newCombinedSeeds+trackExtrapolator+caloTowerForTrk+firstStepPrimaryVerticesUnsorted+ak4CaloJetsForTrk+initialStepTrackRefsForJets+firstStepPrimaryVertices)
fastSim.toReplaceWith(globalreco,_fastSim_globalreco)


globalreco_plusPL= cms.Sequence(globalreco*ctfTracksPixelLess)

reducedRecHits = cms.Sequence ( reducedEcalRecHitsSequence * reducedHcalRecHitsSequence )

highlevelreco = cms.Sequence(egammaHighLevelRecoPrePF*
                             particleFlowReco*
                             egammaHighLevelRecoPostPF*
                             muoncosmichighlevelreco*
                             muonshighlevelreco *
                             particleFlowLinks*
                             jetHighLevelReco*
                             metrecoPlusHCALNoise*
                             btagging*
                             recoPFMET*
                             PFTau*
                             reducedRecHits*
                             cosmicDCTracksSeq
                             )

# AA data with pp reco
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from RecoHI.HiCentralityAlgos.HiCentrality_cfi import hiCentrality
from RecoHI.HiCentralityAlgos.HiClusterCompatibility_cfi import hiClusterCompatibility
_highlevelreco_HI = highlevelreco.copy()
_highlevelreco_HI += hiCentrality
_highlevelreco_HI += hiClusterCompatibility
(pp_on_XeXe_2017 | pp_on_AA_2018).toReplaceWith(highlevelreco, _highlevelreco_HI)
pp_on_AA_2018.toReplaceWith(highlevelreco,highlevelreco.copyAndExclude([PFTau]))

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from RecoHI.HiTracking.HILowPtConformalPixelTracks_cfi import *
_highlevelreco_HI_wPixTracks = highlevelreco.copy()
(pp_on_XeXe_2017 | pp_on_AA_2018).toReplaceWith(highlevelreco, cms.Sequence(_highlevelreco_HI_wPixTracks* hiConformalPixelTracksSequencePhase1))

# not commisoned and not relevant in FastSim (?):
_fastSim_highlevelreco = highlevelreco.copyAndExclude([cosmicDCTracksSeq,muoncosmichighlevelreco])
fastSim.toReplaceWith(highlevelreco,_fastSim_highlevelreco)


from FWCore.Modules.logErrorHarvester_cfi import *

# "Export" Section
reconstruction         = cms.Sequence(localreco*globalreco*highlevelreco*logErrorHarvester)

#logErrorHarvester should only wait for items produced in the reconstruction sequence
_modulesInReconstruction = list()
reconstruction.visit(cms.ModuleNamesFromGlobalsVisitor(globals(),_modulesInReconstruction))
logErrorHarvester.includeModules = cms.untracked.vstring(set(_modulesInReconstruction))

reconstruction_trackingOnly = cms.Sequence(localreco*globalreco_tracking)
reconstruction_pixelTrackingOnly = cms.Sequence(
    pixeltrackerlocalreco*
    offlineBeamSpot*
    siPixelClusterShapeCachePreSplitting*
    recopixelvertexing
)

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
noTrackingAndDependent.append(siPixelRecHitsPreSplitting)
noTrackingAndDependent.append(siStripMatchedRecHits)
noTrackingAndDependent.append(pixelTracks)
noTrackingAndDependent.append(ckftracks)
reconstruction_fromRECO_noTrackingTest = reconstruction.copyAndExclude(modulesToRemove+noTrackingAndDependent)
##requires generalTracks trajectories
noTrackingAndDependent.append(trackerDrivenElectronSeeds)
noTrackingAndDependent.append(ecalDrivenElectronSeeds)
noTrackingAndDependent.append(uncleanedOnlyElectronSeeds)
noTrackingAndDependent.append(uncleanedOnlyElectronCkfTrackCandidates)
noTrackingAndDependent.append(uncleanedOnlyElectronGsfTracks)
noTrackingAndDependent.append(uncleanedOnlyGeneralConversionTrackProducer)
noTrackingAndDependent.append(uncleanedOnlyGsfConversionTrackProducer)
noTrackingAndDependent.append(uncleanedOnlyPfTrackElec)
noTrackingAndDependent.append(uncleanedOnlyGsfElectronCores)
noTrackingAndDependent.append(uncleanedOnlyPfTrack)
noTrackingAndDependent.append(uncleanedOnlyGeneralInOutOutInConversionTrackMerger)#can live with
noTrackingAndDependent.append(uncleanedOnlyGsfGeneralInOutOutInConversionTrackMerger)#can live with
noTrackingAndDependent.append(uncleanedOnlyAllConversions)
noTrackingAndDependent.append(uncleanedOnlyGsfElectrons)#can live with
noTrackingAndDependent.append(electronMergedSeeds)
noTrackingAndDependent.append(electronCkfTrackCandidates)
noTrackingAndDependent.append(electronGsfTracks)
noTrackingAndDependent.append(generalConversionTrackProducer)
noTrackingAndDependent.append(generalInOutOutInConversionTrackMerger)
noTrackingAndDependent.append(gsfGeneralInOutOutInConversionTrackMerger)
noTrackingAndDependent.append(ecalDrivenGsfElectrons)
noTrackingAndDependent.append(gsfConversionTrackProducer)
noTrackingAndDependent.append(allConversions)
noTrackingAndDependent.append(gsfElectrons)
reconstruction_fromRECO_noTracking = reconstruction.copyAndExclude(modulesToRemove+noTrackingAndDependent)
reconstruction_noTracking = reconstruction.copyAndExclude(noTrackingAndDependent)


#sequences with additional stuff
reconstruction_withPixellessTk  = cms.Sequence(localreco        *globalreco_plusPL*highlevelreco*logErrorHarvester)
reconstruction_HcalNZS = cms.Sequence(localreco_HcalNZS*globalreco       *highlevelreco*logErrorHarvester)

#sequences without some stuffs
#
reconstruction_woCosmicMuons = cms.Sequence(localreco*globalreco*highlevelreco*logErrorHarvester)


# define a standard candle. please note I am picking up individual
# modules instead of sequences
#
reconstruction_standard_candle = cms.Sequence(localreco*globalreco*vertexreco*recoJetAssociations*btagging*electronSequence*photonSequence)
