#############################
# This cfg configures the part of reconstruction 
# in FastSim to be done after event mixing
# FastSim mixes tracker information on reconstruction level,
# so tracks are recontructed before mixing.
# At present, only the generalTrack collection, produced with iterative tracking is mixed.
#############################

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.SequenceTypes import _SequenceCollection

# import standard reconstruction
# apply modifications before doing the actual import at the end
import Configuration.StandardSequences.Reconstruction_cff as _reco

# list of modules to be deleted
_mod2del = []

##########################################
# offlineBeamSpot is reconstructed before mixing
########################################## 
_mod2del.append(_reco.offlineBeamSpot)
_reco.globalreco.remove(_reco.offlineBeamSpot) # temporary removing this by hand, cause the usual removal (see end of this file) doesn't seem work

###########################################
# no castor, zdc, Totem RP in FastSim
###########################################
_reco.localreco.remove(_reco.castorreco)
_reco.globalreco.remove(_reco.CastorFullReco)
_reco.hcalLocalRecoSequence.remove(_reco.zdcreco)
_reco.localreco.remove(_reco.totemRPLocalReconstruction)

##########################################
# Calo rechits
##########################################
# not commisoned and not relevant in FastSim (?):
_reco.reducedEcalRecHitsSequence.remove(_reco.seldigis)
_reco.ecalRecHitSequence.remove(_reco.ecalCompactTrigPrim)
_reco.ecalRecHitSequence.remove(_reco.ecalTPSkim)

# no flags for bad channels in FastSim
_reco.ecalRecHit.killDeadChannels = False
_reco.ecalRecHit.recoverEBFE = False
_reco.ecalRecHit.recoverEEFE = False
_reco.ecalUncalibRecHitSequence.remove(_reco.ecalDetIdToBeRecovered)

##########################################
# Changes to tracking sequences
##########################################
# modules to be removed
_mod2del = _reco.trackingGlobalReco.expandAndClone()._seq._collection
_mod2del.append(_reco.trackingGlobalReco)
_mod2del.extend(_reco.recopixelvertexing.expandAndClone()._seq._collection)
_mod2del.append(_reco.MeasurementTrackerEventPreSplitting)
# actually we want to keep a few modules that we need to run (again) after mixing) 
for _entry in [_reco.firstStepPrimaryVertices,_reco.ak4CaloJetsForTrk,_reco.caloTowerForTrk,_reco.trackExtrapolator]:
    while _entry in _mod2del:
        _mod2del.remove(_entry)

# remove tracking sequences from main reco sequences
_reco.localreco.remove(_reco.trackerlocalreco)
_reco.globalreco.remove(_reco.siPixelClusterShapeCachePreSplitting)
_reco.globalreco.remove(_reco.trackingGlobalReco)

# we need a replacment for the firstStepPrimaryVertices
# that includes tracker information of signal and pile up
# after mixing there is no such thing as initialStepTracks,
# so we replace the input collection for firstStepPrimaryVertices with generalTracks
_reco.firstStepPrimaryVertices.TrackLabel = "generalTracks"

# insert the few tracking modules to be run after mixing back in the globalreco sequence
#for _entry in reversed([trackExtrapolator,caloTowerForTrk,firstStepPrimaryVertices,ak4CaloJetsForTrk])
_reco.globalreco.insert(0,_reco.trackExtrapolator+_reco.caloTowerForTrk+_reco.firstStepPrimaryVertices+_reco.ak4CaloJetsForTrk)

# FastSim doesn't use Runge Kute for propagation
# the following propagators are not used in FastSim, but just to be sure...
_reco.KFFitterForRefitOutsideIn.Propagator = 'SmartPropagatorAny'
_reco.KFSmootherForRefitOutsideIn.Propagator = 'SmartPropagator'

# replace the standard ecal-driven seeds with the FastSim emulated ones
import FastSimulation.Tracking.ElectronSeeds_cff
_reco.newCombinedSeeds = FastSimulation.Tracking.ElectronSeeds_cff.newCombinedSeeds
_reco.globalreco.insert(0,_reco.newCombinedSeeds)

##########################################
# FastSim changes to electron reconstruction
##########################################
# tracker driven electron seeds depend on the generalTracks trajectory collection
# However, in FastSim jobs, trajectories are only available for the 'before mixing' track collections
# Therefore we let the seeds depend on the 'before mixing' generalTracks collection
# TODO: investigate whether the dependence on trajectories can be avoided
import FastSimulation.Tracking.ElectronSeedTrackRefFix_cfi
_trackerDrivenElectronSeeds = FastSimulation.Tracking.ElectronSeedTrackRefFix_cfi.fixedTrackerDrivenElectronSeeds.clone()
_reco.electronSeeds.replace(_reco.trackerDrivenElectronSeeds,_reco.trackerDrivenElectronSeeds+_trackerDrivenElectronSeeds)
_reco.trackerDrivenElectronSeedsTmp = _reco.trackerDrivenElectronSeeds
_reco.trackerDrivenElectronSeedsTmp.TkColList = cms.VInputTag(cms.InputTag("generalTracksBeforeMixing"))
_reco.trackerDrivenElectronSeeds = _trackerDrivenElectronSeeds
_reco.trackerDrivenElectronSeeds.seedCollection.setModuleLabel("trackerDrivenElectronSeedsTmp") 
_reco.trackerDrivenElectronSeeds.idCollection.setModuleLabel("trackerDrivenElectronSeedsTmp")

# replace the ECAL driven electron track candidates with the FastSim emulated ones
import FastSimulation.Tracking.electronCkfTrackCandidates_cff
_reco.fastElectronCkfTrackCandidates = FastSimulation.Tracking.electronCkfTrackCandidates_cff.electronCkfTrackCandidates.clone()
_reco.electronGsfTracking.replace(_reco.electronCkfTrackCandidates,_reco.fastElectronCkfTrackCandidates)
_reco.electronGsfTracks.src = "fastElectronCkfTrackCandidates"

# FastSim has no template fit on tracker hits
_reco.electronGsfTracks.TTRHBuilder = "WithoutRefit"

# the conversion producer depends on trajectories
# so we feed it with the 'before mixing' track colletion
_reco.generalConversionTrackProducer.TrackProducer = 'generalTracksBeforeMixing'

# then we need to fix the track references, so that they point to the final track collection, after mixing
import FastSimulation.Tracking.ConversionTrackRefFix_cfi
_conversionTrackRefFix = FastSimulation.Tracking.ConversionTrackRefFix_cfi.fixedConversionTracks.clone(
    src = cms.InputTag("generalConversionTrackProducerTmp"))
_reco.conversionTrackSequenceNoEcalSeeded.replace(_reco.generalConversionTrackProducer,_reco.generalConversionTrackProducer+_conversionTrackRefFix)
_reco.generalConversionTrackProducerTmp = _reco.generalConversionTrackProducer
_reco.generalConversionTrackProducer = _conversionTrackRefFix

# this might be historical: not sure why we do this
_reco.egammaGlobalReco.replace(_reco.conversionTrackSequence,_reco.conversionTrackSequenceNoEcalSeeded)
_reco.allConversions.src = 'gsfGeneralConversionTrackMerger'
# TODO: revisit converions in FastSim

# not commisoned and not relevant in FastSim (?):
_reco.egammaHighLevelRecoPrePF.remove(_reco.uncleanedOnlyElectronSequence)

# not commisoned and not relevant in FastSim (?):
_reco.egammareco.remove(_reco.conversionSequence)
_reco.egammaHighLevelRecoPrePF.remove(_reco.conversionSequence)

##########################################
# FastSim changes to muon reconstruction
##########################################
# not commisoned and not relevant in FastSim (?):
_reco.globalreco.remove(_reco.muoncosmicreco)
_reco.highlevelreco.remove(_reco.cosmicDCTracksSeq)
_reco.highlevelreco.remove(_reco.muoncosmichighlevelreco)
_reco.muons.FillCosmicsIdMap = False

# not commisoned and not relevant in FastSim (?):
_reco.globalmuontracking.remove(_reco.displacedGlobalMuonTracking)
_reco.standalonemuontracking.remove(_reco.displacedMuonSeeds)
_reco.standalonemuontracking.remove(_reco.displacedStandAloneMuons)

# not commisoned and not relevant in FastSim (?):
_reco.muonGlobalReco.remove(_reco.muonreco_with_SET)

# not commisoned and not relevant in FastSim (?):
_reco.muonGlobalReco.remove(_reco.muonSelectionTypeSequence)
_reco.muons.FillSelectorMaps = False

# FastSim has no template fit on tracker hits
_reco.globalMuons.GLBTrajBuilderParameters.GlbRefitterParameters.TrackerRecHitBuilder = 'WithoutRefit'
_reco.globalMuons.GLBTrajBuilderParameters.TrackerRecHitBuilder = 'WithoutRefit'
_reco.globalMuons.GLBTrajBuilderParameters.TrackTransformer.TrackerRecHitBuilder = 'WithoutRefit'
_reco.tevMuons.RefitterParameters.TrackerRecHitBuilder = 'WithoutRefit'

# FastSim doesn't use Runge Kute for propagation
_reco.globalMuons.GLBTrajBuilderParameters.GlbRefitterParameters.Propagator = 'SmartPropagatorAny'
_reco.globalMuons.GLBTrajBuilderParameters.GlobalMuonTrackMatcher.Propagator = 'SmartPropagator'
_reco.globalMuons.GLBTrajBuilderParameters.TrackTransformer.Propagator = 'SmartPropagatorAny'
_reco.GlbMuKFFitter.Propagator = "SmartPropagatorAny"
_reco.GlobalMuonRefitter.Propagator = "SmartPropagatorAny"
_reco.KFSmootherForMuonTrackLoader.Propagator = "SmartPropagatorAny"
_reco.KFSmootherForRefitInsideOut.Propagator = "SmartPropagatorAny"
_reco.KFFitterForRefitInsideOut.Propagator = "SmartPropagatorAny"
_reco.tevMuons.RefitterParameters.Propagator = "SmartPropagatorAny"

##########################################
# FastSim changes to jet/met reconstruction
##########################################
# CSCHaloData depends on cosmic muons, not available in fastsim
_reco.BeamHaloId.remove(_reco.CSCHaloData)
# GlobalHaloData and BeamHaloSummary depend on CSCHaloData
_reco.BeamHaloId.remove(_reco.GlobalHaloData)
_reco.BeamHaloId.remove(_reco.BeamHaloSummary)

############################################
# deleting modules to avoid clashes with part of reconstruction run before mixing
############################################
for _entry in _mod2del:
    for _key,_value in _reco.__dict__.items():
        _index = -1
        if isinstance(_value,cms.Sequence):
            try:
                _index = _value.index(_entry)
            except:
                pass
        if _index >= 0:
            _value.remove(_entry)
            # removing the last item does not work, and changes the properties of the sequence
            # so, we detect this changed behaviour and add the sequence to the list of items to be deleted
            # this is buggy
            if not isinstance(_value._seq,_SequenceCollection):
                _mod2del.append(_value)

# and delete
for _entry in _mod2del:
    for _key,_value in _reco.__dict__.items():
        if _entry == _value:
            delattr(_reco,_key)

############################################
# the actual import
############################################
from Configuration.StandardSequences.Reconstruction_cff import *

