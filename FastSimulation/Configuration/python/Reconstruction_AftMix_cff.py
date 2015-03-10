import FWCore.ParameterSet.Config as cms
from FastSimulation.Configuration.CfgUtilities import *

from Configuration.StandardSequences.Reconstruction_cff import *

###########################################
# no castor in FastSim
###########################################
localreco.remove(castorreco)
globalreco.remove(CastorFullReco)


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
namesObjsToDel = set(getSeqEntryNames(trackerlocalreco,locals()))
namesObjsToDel = namesObjsToDel.union(getSeqEntryNames(trackingGlobalReco,locals()))
namesObjsToDel = namesObjsToDel.union(getSeqEntryNames(recopixelvertexing,locals()))
# actually we want to keep a few modules that we need to run (again) after mixing)
namesObjsToDel = namesObjsToDel.difference(["firstStepPrimaryVertices","ak4CaloJetsForTrk","caloTowerForTrk","trackExtrapolator"])

# remove the content of the following sequences from every sequence
removeSeqEntriesEveryWhere(trackerlocalreco,locals())
removeSeqEntriesEveryWhere(trackingGlobalReco,locals())
removeSeqEntriesEveryWhere(recopixelvertexing,locals())

# now do the actual deletion
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
trackerDrivenElectronSeeds.TkColList = ['generalTracksBeforeMixing']
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


##########################################
# FastSim changes to muon reconstruction
##########################################
# because there is no initialStep track collection with both pu and signal tracks:
earlyDisplacedMuons.inputCollectionLabels[0] = "generalTracks"

# no veto on cosmic muons because it relies on track finding
removeSeqEntriesEveryWhere(muoncosmichighlevelreco,locals())
muons.FillCosmicsIdMap = False

# FastSim has no template fit on tracker hits
GlobalMuonRefitter.TrackerRecHitBuilder = 'WithoutRefit'
tevMuons.RefitterParameters.TrackerRecHitBuilder = 'WithoutRefit'

# FastSim uses a faster propagator
GlbMuKFFitter.Propagator = "SmartPropagatorAny"
GlobalMuonRefitter.Propagator = "SmartPropagatorAny"
KFSmootherForMuonTrackLoader.Propagator = "SmartPropagatorAny"
KFSmootherForRefitInsideOut.Propagator = "SmartPropagatorAny"
KFFitterForRefitInsideOut.Propagator = "SmartPropagatorAny"
tevMuons.RefitterParameters.Propagator = "SmartPropagatorAny"

##########################################
# FastSim changes to jet/met reconstruction
##########################################
# because it crashes and is not used in FastSim samples as far as we know
jetHighLevelReco.remove(recoJetAssociationsExplicit)

# depends on cosmic muons reco, which doesn't run in FastSim
metreco.remove(BeamHaloId)

##########################################
# The PF Patch
##########################################
# throws random neutral hadrons in the detector to fix jet response
# we should get rid of this asap, but the source of the issue is not pointed down
particleFlowTmpTmp = particleFlowTmp
from FastSimulation.ParticleFlow.FSparticleFlow_cfi import FSparticleFlow as particleFlowTmp
particleFlowTmp.pfCandidates = cms.InputTag("particleFlowTmpTmp")
particleFlowReco.insert(particleFlowReco.index(particleFlowTmpTmp)+1,particleFlowTmp)

###########################################
# sequences that are not part of any path
# but cause python compilation errors
###########################################
del ckftracks_plus_pixelless
del ckftracks_woBH
del reconstruction_fromRECO
del ckftracks_wodEdX

############################################
# the final reconstruction sequence
############################################
reconstruction = cms.Sequence(localreco*newCombinedSeeds*globalreco*highlevelreco*logErrorHarvester)
