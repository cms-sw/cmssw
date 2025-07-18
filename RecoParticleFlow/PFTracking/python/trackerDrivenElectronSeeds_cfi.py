import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.modules import GoodSeedProducer
trackerDrivenElectronSeeds = GoodSeedProducer()

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
for e in [pp_on_XeXe_2017, pp_on_AA]:
    e.toModify(trackerDrivenElectronSeeds, MinPt = 5.0) 

# tracker driven electron seeds depend on the generalTracks trajectory collection
# However, in FastSim jobs, trajectories are only available for the 'before mixing' track collections
# Therefore we let the seeds depend on the 'before mixing' generalTracks collection
# TODO: investigate whether the dependence on trajectories can be avoided
from Configuration.Eras.Modifier_fastSim_cff import fastSim
trackerDrivenElectronSeedsTmp = trackerDrivenElectronSeeds.clone(TkColList = ["generalTracksBeforeMixing"])
import FastSimulation.Tracking.ElectronSeedTrackRefFix_cfi
_fastSim_trackerDrivenElectronSeeds = FastSimulation.Tracking.ElectronSeedTrackRefFix_cfi.fixedTrackerDrivenElectronSeeds.clone()
_fastSim_trackerDrivenElectronSeeds.seedCollection.setModuleLabel("trackerDrivenElectronSeedsTmp")
_fastSim_trackerDrivenElectronSeeds.idCollection = ["trackerDrivenElectronSeedsTmp:preid",]
fastSim.toReplaceWith(trackerDrivenElectronSeeds,_fastSim_trackerDrivenElectronSeeds)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(trackerDrivenElectronSeeds,MinPt = 1.0)

