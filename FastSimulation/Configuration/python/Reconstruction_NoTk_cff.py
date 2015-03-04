import FWCore.ParameterSet.Config as cms
from FastSimulation.Configuration.CfgUtilities import *

from Configuration.StandardSequences.Reconstruction_cff import *

_firstStepPrimaryVertices = firstStepPrimaryVertices.clone(
    TrackLabel = "generalTracks"
)
_ak4CaloJetsForTrk = ak4CaloJetsForTrk.clone()
_caloTowerForTrk = caloTowerForTrk.clone()

# in FastSim, tracking is performed before mixing,
# therefore it is kept outside the usual reconstruction sequence
localreco.remove(trackerlocalreco)
globalreco.remove(MeasurementTrackerEventPreSplitting)
globalreco.remove(siPixelClusterShapeCachePreSplitting)
globalreco.remove(trackingGlobalReco)

# FastSim has no castor
localreco.remove(castorreco)
globalreco.remove(CastorFullReco)


# delete modules that have different definition in FastSim
# need full deletion, else process.load("FastSimulation.Configuration.Reconstruction_NoTk_cff")
# complaints about objects with same name loaded in process.load("FastSimulation.Configuration.Reconstruction_Tk_cff.py")
namesObjsToDel = getSeqEntryNames(trackerlocalreco,locals())
namesObjsToDel.extend(getSeqEntryNames(trackingGlobalReco,locals()))
namesObjsToDel.extend(getSeqEntryNames(recopixelvertexing,locals()))
removeSeqEntriesEveryWhere(trackerlocalreco,locals())
removeSeqEntriesEveryWhere(trackingGlobalReco,locals())
removeSeqEntriesEveryWhere(recopixelvertexing,locals())
for name in namesObjsToDel:
    exec("del " + name)

# beamspot before mixing
globalreco.remove(offlineBeamSpot)
del offlineBeamSpot

ak4CaloJetsForTrk = _ak4CaloJetsForTrk
globalreco.insert(0,ak4CaloJetsForTrk)

firstStepPrimaryVertices = _firstStepPrimaryVertices
globalreco.insert(0,firstStepPrimaryVertices)

caloTowerForTrk = _caloTowerForTrk
globalreco.insert(0,caloTowerForTrk)

# put track extrapoloation back, cause it has to be done after mixing,
# on both signal and bkg tracks
from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import trackExtrapolator
globalreco.insert(0,trackExtrapolator)


generalConversionTrackProducer.TrackProducer = 'generalTracksBeforeMixing'
trackerDrivenElectronSeeds.TkColList = ['generalTracksBeforeMixing']

from FastSimulation.Tracking.globalCombinedSeeds_cfi import newCombinedSeeds

egammaGlobalReco.replace(conversionTrackSequence,conversionTrackSequenceNoEcalSeeded)
allConversions.src = 'gsfGeneralConversionTrackMerger'

earlyDisplacedMuons.inputCollectionLabels[0] = "generalTracks"

# seems not to be used and causes troubles with FastSim
egammaHighLevelRecoPrePF.remove(uncleanedOnlyElectronSequence)

# because it crashes and is not used (?)
jetHighLevelReco.remove(recoJetAssociationsExplicit)

reconstruction = cms.Sequence(localreco*newCombinedSeeds*globalreco*highlevelreco*logErrorHarvester)
