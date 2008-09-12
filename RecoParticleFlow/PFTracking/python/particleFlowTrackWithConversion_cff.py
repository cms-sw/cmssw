import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.elecPreId_cff import *
from RecoParticleFlow.PFTracking.gsfSeedClean_cfi import *
from TrackingTools.GsfTracking.CkfElectronCandidates_cff import *
from TrackingTools.GsfTracking.GsfElectrons_cff import *
from RecoParticleFlow.PFTracking.pfNuclear_cfi import *
from RecoEgamma.EgammaElectronProducers.gsfElectronCkfTrackCandidateMaker_cff import *
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
gsfElCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import TrackingTools.GsfTracking.GsfElectronFit_cfi
gsfPFtracks = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
from RecoEgamma.EgammaPhotonProducers.softConversionSequence_cff import *
from RecoParticleFlow.PFTracking.pfConversions_cfi import *

#TRAJECTORIES IN THE EVENT
softConversionIOTracks.TrajectoryInEvent = cms.bool(True)
softConversionOITracks.TrajectoryInEvent = cms.bool(True)

#UNCOMMENT THE LINES THAT START WITH #UN# IN ORDER TO ADD CONVERSION FROM PF CLUSTERS

#UN#pfConversions.OtherConversionCollection =cms.VInputTag(cms.InputTag("softConversions:softConversionCollection"))
#UN#pfConversions.OtherOutInCollection      =           cms.VInputTag(cms.InputTag("softConversionOITracks"))
#UN#pfConversions.OtherInOutCollection      =           cms.VInputTag(cms.InputTag("softConversionIOTracks"))

particleFlowTrackWithConversion =cms.Sequence(
    elecPreId*
    gsfSeedclean*
    gsfElCandidates*
    gsfPFtracks*
    pfTrackElec*
#UN#    softConversionSequence*
    pfConversions
    )


gsfElCandidates.TrajectoryBuilder = 'TrajectoryBuilderForPixelMatchGsfElectrons'
gsfElCandidates.SeedProducer = 'gsfSeedclean'
gsfElCandidates.SeedLabel = ''
gsfPFtracks.Fitter = 'GsfElectronFittingSmoother'
gsfPFtracks.Propagator = 'fwdElectronPropagator'
gsfPFtracks.src = 'gsfElCandidates'
gsfPFtracks.TTRHBuilder = 'WithTrackAngle'
gsfPFtracks.TrajectoryInEvent = True


