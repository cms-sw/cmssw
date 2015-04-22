#############################
# This cfg configures the part of reconstruction 
# in FastSim to be done after event mixing
# FastSim mixes tracker information on reconstruction level,
# so tracks are recontructed before mixing.
# At present, only the generalTrack collection, produced with iterative tracking is mixed.
#############################

import FWCore.ParameterSet.Config as cms

###########################################
# remove modules from Reconstruction_cff that are run before mixing
###########################################

import Configuration.StandardSequences.Reconstruction_cff as _reco

# list the modules we want to get rid of
_mod2del = _reco.trackingGlobalReco.expandAndClone()._seq._collection
_mod2del.extend(_reco.recopixelvertexing.expandAndClone()._seq._collection)
_mod2del = set(_mod2del)                                                                                                                                                                                   

# actually we want to keep a few modules that we need to run (again) after mixing) 
_mod2del = _mod2del.difference([_reco.firstStepPrimaryVertices,_reco.ak4CaloJetsForTrk,_reco.caloTowerForTrk,_reco.trackExtrapolator,_reco.newCombinedSeeds])

# and there are a few extra ones we want to get rid off

# offlineBeamSpot is reconstructed before mixing 
_mod2del.add(_reco.offlineBeamSpot)

# get rid of those modules
for _key,_value in _reco.__dict__.items():
    for _entry in _mod2del:
        # remove from all sequences
        try:
            _value.remove(_entry)
        except:
            pass
        # and delete
        if _entry == _value:
            delattr(_reco,_key)


###########################################
# import the standard reconstruction sequences
###########################################
fragment = cms.ProcessFragment("RECO")
fragment.load("Configuration.StandardSequences.Reconstruction_cff")

###########################################
# no castor / zdc in FastSim
###########################################
fragment.localreco.remove(fragment.castorreco)
fragment.globalreco.remove(fragment.CastorFullReco)
fragment.hcalLocalRecoSequence.remove(fragment.zdcreco)

##########################################
# Calo rechits
##########################################

# not commisoned and not relevant in FastSim (?):
fragment.reducedEcalRecHitsSequence.remove(fragment.seldigis)
fragment.ecalRecHitSequence.remove(fragment.ecalCompactTrigPrim)
fragment.ecalRecHitSequence.remove(fragment.ecalTPSkim)

# no flags for bad channels in FastSim
fragment.ecalRecHit.killDeadChannels = False
fragment.ecalRecHit.recoverEBFE = False
fragment.ecalRecHit.recoverEEFE = False
fragment.ecalUncalibRecHitSequence.remove(fragment.ecalDetIdToBeRecovered)
##########################################
# Changes to tracking sequences
##########################################

# remove tracking 
fragment.localreco.remove(fragment.trackerlocalreco)
fragment.globalreco.remove(fragment.MeasurementTrackerEventPreSplitting)
fragment.globalreco.remove(fragment.siPixelClusterShapeCachePreSplitting)
fragment.globalreco.remove(fragment.trackingGlobalReco)


# we need a replacment for the firstStepPrimaryVertices
# that includes tracker information of signal and pile up
# after mixing there is no such thing as initialStepTracks,
# so we replace the input collection for firstStepPrimaryVertices with generalTracks
fragment.firstStepPrimaryVertices.TrackLabel = "generalTracks"

# insert the few tracking modules to be run after mixing back in the globalreco sequence
#for _entry in reversed([trackExtrapolator,caloTowerForTrk,firstStepPrimaryVertices,ak4CaloJetsForTrk])
fragment.globalreco.insert(0,fragment.trackExtrapolator+fragment.caloTowerForTrk+fragment.firstStepPrimaryVertices+fragment.ak4CaloJetsForTrk+fragment.newCombinedSeeds)

# FastSim doesn't use Runge Kute for propagation
# the following propagators are not used in FastSim, but just to be sure...
fragment.KFFitterForRefitOutsideIn.Propagator = 'SmartPropagatorAny'
fragment.KFSmootherForRefitOutsideIn.Propagator = 'SmartPropagator'


##########################################
# FastSim changes to electron reconstruction
##########################################

# replace the standard ecal-driven seeds with the FastSim emulated ones
import FastSimulation.Tracking.globalCombinedSeeds_cfi
fragment.newCombinedSeeds = FastSimulation.Tracking.globalCombinedSeeds_cfi.newCombinedSeeds


# tracker driven electron seeds depend on the generalTracks trajectory collection
# However, in FastSim jobs, trajectories are only available for the 'before mixing' track collections
# Therefore we let the seeds depend on the 'before mixing' generalTracks collection
# TODO: investigate whether the dependence on trajectories can be avoided
fragment.trackerDrivenElectronSeedsTmp = fragment.trackerDrivenElectronSeeds.clone(
    TkColList = cms.VInputTag(cms.InputTag("generalTracksBeforeMixing")))
import FastSimulation.Tracking.ElectronSeedTrackRefFix_cfi
fragment.trackerDrivenElectronSeeds = FastSimulation.Tracking.ElectronSeedTrackRefFix_cfi.fixedTrackerDrivenElectronSeeds.clone()
fragment.trackerDrivenElectronSeeds.seedCollection.setModuleLabel("trackerDrivenElectronSeedsTmp") 
fragment.trackerDrivenElectronSeeds.idCollection.setModuleLabel("trackerDrivenElectronSeedsTmp")
fragment.electronSeeds.replace(fragment.trackerDrivenElectronSeeds,fragment.trackerDrivenElectronSeedsTmp+fragment.trackerDrivenElectronSeeds)

# replace the ECAL driven electron track candidates with the FastSim emulated ones
import FastSimulation.EgammaElectronAlgos.electronGSGsfTrackCandidates_cff
fragment.electronGSGsfTrackCandidates = FastSimulation.EgammaElectronAlgos.electronGSGsfTrackCandidates_cff.electronGSGsfTrackCandidates
fragment.electronGsfTracking.replace(fragment.electronCkfTrackCandidates,fragment.electronGSGsfTrackCandidates)
fragment.electronGsfTracks.src = "electronGSGsfTrackCandidates"

# FastSim has no template fit on tracker hits
fragment.electronGsfTracks.TTRHBuilder = "WithoutRefit"

# the conversion producer depends on trajectories
# so we feed it with the 'before mixing' track colletion
fragment.generalConversionTrackProducer.TrackProducer = 'generalTracksBeforeMixing'

# this might be historical: not sure why we do this
fragment.egammaGlobalReco.replace(fragment.conversionTrackSequence,fragment.conversionTrackSequenceNoEcalSeeded)
fragment.allConversions.src = 'gsfGeneralConversionTrackMerger'
# TODO: revisit converions in FastSim

# not commisoned and not relevant in FastSim (?):
fragment.egammaHighLevelRecoPrePF.remove(fragment.uncleanedOnlyElectronSequence)

# not commisoned and not relevant in FastSim (?):
fragment.egammareco.remove(fragment.conversionSequence)
fragment.egammaHighLevelRecoPrePF.remove(fragment.conversionSequence)


##########################################
# FastSim changes to muon reconstruction
##########################################
# not commisoned and not relevant in FastSim (?):
fragment.globalreco.remove(fragment.muoncosmicreco)
fragment.highlevelreco.remove(fragment.muoncosmichighlevelreco)
fragment.muons.FillCosmicsIdMap = False

# not commisoned and not relevant in FastSim (?):
fragment.globalmuontracking.remove(fragment.displacedGlobalMuonTracking)
fragment.standalonemuontracking.remove(fragment.displacedMuonSeeds)
fragment.standalonemuontracking.remove(fragment.displacedStandAloneMuons)

# not commisoned and not relevant in FastSim (?):
fragment.muonGlobalReco.remove(fragment.muonreco_with_SET)

# not commisoned and not relevant in FastSim (?):
fragment.muonGlobalReco.remove(fragment.muonSelectionTypeSequence)
fragment.muons.FillSelectorMaps = False

# FastSim has no template fit on tracker hits
fragment.globalMuons.GLBTrajBuilderParameters.GlbRefitterParameters.TrackerRecHitBuilder = 'WithoutRefit'
fragment.globalMuons.GLBTrajBuilderParameters.TrackerRecHitBuilder = 'WithoutRefit'
fragment.globalMuons.GLBTrajBuilderParameters.TrackTransformer.TrackerRecHitBuilder = 'WithoutRefit'
fragment.tevMuons.RefitterParameters.TrackerRecHitBuilder = 'WithoutRefit'

# FastSim doesn't use Runge Kute for propagation
fragment.globalMuons.GLBTrajBuilderParameters.GlbRefitterParameters.Propagator = 'SmartPropagatorAny'
fragment.globalMuons.GLBTrajBuilderParameters.GlobalMuonTrackMatcher.Propagator = 'SmartPropagator'
fragment.globalMuons.GLBTrajBuilderParameters.TrackTransformer.Propagator = 'SmartPropagatorAny'
fragment.GlbMuKFFitter.Propagator = "SmartPropagatorAny"
fragment.GlobalMuonRefitter.Propagator = "SmartPropagatorAny"
fragment.KFSmootherForMuonTrackLoader.Propagator = "SmartPropagatorAny"
fragment.KFSmootherForRefitInsideOut.Propagator = "SmartPropagatorAny"
fragment.KFFitterForRefitInsideOut.Propagator = "SmartPropagatorAny"
fragment.tevMuons.RefitterParameters.Propagator = "SmartPropagatorAny"


##########################################
# FastSim changes to jet/met reconstruction
##########################################

# not commisoned and not relevant in FastSim (?):
fragment.jetHighLevelReco.remove(fragment.recoJetAssociationsExplicit)

# not commisoned and not relevant in FastSim (?):
fragment.metreco.remove(fragment.BeamHaloId)

# not commisoned and not relevant in FastSim (?):
fragment.metrecoPlusHCALNoise.remove(fragment.hcalnoise)

##########################################
# The PF Patch
##########################################

# throws random neutral hadrons in the detector to fix jet response
# we should get rid of this asap, but the source of the issue is not known
fragment.particleFlowTmpTmp = fragment.particleFlowTmp.clone()
import FastSimulation.ParticleFlow.FSparticleFlow_cfi
fragment.particleFlowTmp = FastSimulation.ParticleFlow.FSparticleFlow_cfi.FSparticleFlow
fragment.particleFlowTmp.pfCandidates = cms.InputTag("particleFlowTmpTmp")
fragment.particleFlowReco.replace(fragment.particleFlowTmp,fragment.particleFlowTmpTmp+fragment.particleFlowTmp)
