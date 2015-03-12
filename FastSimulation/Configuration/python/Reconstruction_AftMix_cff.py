#############################
# This cfg configures the part of reconstruction 
# in FastSim to be done after event mixing
# FastSim mixes tracker information on reconstruction level,
# so tracks are recontructed before mixing.
# At present, only the generalTrack collection, produced with iterative tracking is mixed.
#############################

import FWCore.ParameterSet.Config as cms

###########################################
# import the standard reconstruction sequences
###########################################
from Configuration.StandardSequences.Reconstruction_cff import *


###########################################
# no castor / zdc in FastSim
###########################################
localreco.remove(castorreco)
globalreco.remove(CastorFullReco)
hcalLocalRecoSequence.remove(zdcreco)

###########################################
# FastSim does the beamspot before mixing
###########################################
globalreco.remove(offlineBeamSpot)
del offlineBeamSpot


##########################################
# remove most of the tracking , since it is run before mixing
##########################################

# remove tracking 
localreco.remove(trackerlocalreco)
globalreco.remove(MeasurementTrackerEventPreSplitting)
globalreco.remove(siPixelClusterShapeCachePreSplitting)
globalreco.remove(trackingGlobalReco)

# list names of tracking modules to be deleted
# weneed a full deletion to avoid interference with fast tracking
# that uses the same names as full tracking for corresponding collections
from FastSimulation.Configuration.CfgUtilities import getSeqEntryNames,removeSeqEntriesEveryWhere
namesObjsToDel = set(getSeqEntryNames(trackerlocalreco,locals()))
namesObjsToDel = namesObjsToDel.union(getSeqEntryNames(trackingGlobalReco,locals()))
namesObjsToDel = namesObjsToDel.union(getSeqEntryNames(recopixelvertexing,locals()))
# actually we want to keep a few modules that we need to run (again) after mixing)
namesObjsToDel = namesObjsToDel.difference(["firstStepPrimaryVertices","ak4CaloJetsForTrk","caloTowerForTrk","trackExtrapolator"])

# remove the content of the following sequences from every sequence
removeSeqEntriesEveryWhere(trackerlocalreco,locals())
removeSeqEntriesEveryWhere(trackingGlobalReco,locals())
removeSeqEntriesEveryWhere(recopixelvertexing,locals())

# now do the actual deletions
for name in namesObjsToDel:
    exec("del " + name)

# we need a replacment for the firstStepPrimaryVertices
# that includes tracker information of signal and pile up
# after mixing there is no such thing as initialStepTracks,
# so we simple reconstruct the firstStepPrimaryVertices from all tracks
_firstStepPrimaryVertices = firstStepPrimaryVertices.clone(
    TrackLabel = "generalTracks"
)

# insert the few tracking modules to be run after mixing back in the globalreco sequence
globalreco.insert(0,ak4CaloJetsForTrk)
globalreco.insert(0,firstStepPrimaryVertices)
globalreco.insert(0,caloTowerForTrk)
globalreco.insert(0,trackExtrapolator)

##########################################
# FastSim changes to electron reconstruction
##########################################
# ecal-driven seeds are emulated in FastSim
from FastSimulation.Tracking.globalCombinedSeeds_cfi import newCombinedSeeds

# tracker driven electron seeds depend on the generalTracks trajectory collection
# However, in FastSim jobs, trajectories are only available for signal tracks
# Therefore we let the seeds depend on the 'before mixing' generalTracks collection
trackerDrivenElectronSeeds.TkColList = [cms.InputTag('generalTracksBeforeMixing')]
# TODO: add the fixes to the TrackRefs in the tracker-driven seeds

# the conversion producer depends on trajectories
# same reasoning as for tracker driven electron seeds
generalConversionTrackProducer.TrackProducer = 'generalTracksBeforeMixing'

# why was that?
egammaGlobalReco.replace(conversionTrackSequence,conversionTrackSequenceNoEcalSeeded)
allConversions.src = 'gsfGeneralConversionTrackMerger'

# FastSim emulates track finding for electrons
from FastSimulation.EgammaElectronAlgos.electronGSGsfTrackCandidates_cff import electronGSGsfTrackCandidates
electronGsfTracking.replace(electronCkfTrackCandidates,electronGSGsfTrackCandidates)
del electronCkfTrackCandidates
electronGsfTracks.src = "electronGSGsfTrackCandidates"

# FastSim has no template fit on tracker hits
electronGsfTracks.TTRHBuilder = "WithoutRefit"

# seems not to be used and causes crashes in FastSim
egammaHighLevelRecoPrePF.remove(uncleanedOnlyElectronSequence)

# having this in FastSim requires significant work
# all related tracking needs to be emulated
egammareco.remove(conversionSequence)
egammaHighLevelRecoPrePF.remove(conversionSequence)

# wasn't there before and is not used (?)

ecalRecHitSequence.remove(ecalTPSkim)

##########################################
# FastSim changes to muon reconstruction
##########################################

# not commisoned and not relevant in FastSim (?):
globalreco.remove(muoncosmicreco)
highlevelreco.remove(muoncosmichighlevelreco)
muons.FillCosmicsIdMap = False

# not commisoned and not relevant in FastSim (?):
globalmuontracking.remove(displacedGlobalMuonTracking)
standalonemuontracking.remove(displacedMuonSeeds)
standalonemuontracking.remove(displacedStandAloneMuons)

# not commisoned and not relevant in FastSim (?):
muonGlobalReco.remove(muonreco_with_SET)

# not commisoned and not relevant in FastSim (?):
muonGlobalReco.remove(muonSelectionTypeSequence)
muons.FillSelectorMaps = False

# FastSim has no template fit on tracker hits
globalMuons.GLBTrajBuilderParameters.GlbRefitterParameters.TrackerRecHitBuilder = 'WithoutRefit'
globalMuons.GLBTrajBuilderParameters.TrackerRecHitBuilder = 'WithoutRefit'
globalMuons.GLBTrajBuilderParameters.TrackTransformer.TrackerRecHitBuilder = 'WithoutRefit'
tevMuons.RefitterParameters.TrackerRecHitBuilder = 'WithoutRefit'

# FastSim doesn't use Runge Kute for propagation
globalMuons.GLBTrajBuilderParameters.GlbRefitterParameters.Propagator = 'SmartPropagatorAny'
globalMuons.GLBTrajBuilderParameters.GlobalMuonTrackMatcher.Propagator = 'SmartPropagator'
globalMuons.GLBTrajBuilderParameters.TrackTransformer.Propagator = 'SmartPropagatorAny'
GlbMuKFFitter.Propagator = "SmartPropagatorAny"
GlobalMuonRefitter.Propagator = "SmartPropagatorAny"
KFSmootherForMuonTrackLoader.Propagator = "SmartPropagatorAny"
KFSmootherForRefitInsideOut.Propagator = "SmartPropagatorAny"
KFFitterForRefitInsideOut.Propagator = "SmartPropagatorAny"
tevMuons.RefitterParameters.Propagator = "SmartPropagatorAny"


##########################################
# FastSim changes to jet/met reconstruction
##########################################

# not commisoned and not relevant in FastSim (?):
jetHighLevelReco.remove(recoJetAssociationsExplicit)

# not commisoned and not relevant in FastSim (?):
metreco.remove(BeamHaloId)

# not commisoned and not relevant in FastSim (?):
metrecoPlusHCALNoise.remove(hcalnoise)

##########################################
# The PF Patch
##########################################

# throws random neutral hadrons in the detector to fix jet response
# we should get rid of this asap, but the source of the issue is not pointed down
particleFlowTmpTmp = particleFlowTmp
from FastSimulation.ParticleFlow.FSparticleFlow_cfi import FSparticleFlow as particleFlowTmp
particleFlowTmp.pfCandidates = cms.InputTag("particleFlowTmpTmp")
particleFlowReco.insert(particleFlowReco.index(particleFlowTmpTmp)+1,particleFlowTmp)


##########################################
# Calo rechits
##########################################

# not commisoned and not relevant in FastSim (?):
reducedEcalRecHitsSequence.remove(seldigis)
ecalRecHitSequence.remove(ecalCompactTrigPrim)


###########################################
# sequences that are not part of any path
# but cause python compilation errors after the full track sequence removal
###########################################

del ckftracks_plus_pixelless
del ckftracks_woBH
del reconstruction_fromRECO
del ckftracks_wodEdX

###########################################
# deleting some services that are not used
###########################################

del BeamHaloMPropagatorAlong
del BeamHaloMPropagatorOpposite
del BeamHaloPropagatorAlong
del BeamHaloPropagatorAny
del BeamHaloPropagatorOpposite
del BeamHaloSHPropagatorAlong
del BeamHaloSHPropagatorAny
del BeamHaloSHPropagatorOpposite

############################################
# the final reconstruction sequence
############################################
reconstruction = cms.Sequence(localreco*newCombinedSeeds*globalreco*highlevelreco*logErrorHarvester)
