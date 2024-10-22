import FWCore.ParameterSet.Config as cms
from CalibTracker.SiPixelLorentzAngle.SiPixelLorentzAnglePCLWorker_cfi import SiPixelLorentzAnglePCLWorker as worker
from CalibTracker.SiPixelLorentzAngle.siPixelLorentzAnglePCLHarvester_cfi import siPixelLorentzAnglePCLHarvester as _defaultHarvester

SiPixelLorentzAnglePCLHarvester = _defaultHarvester.clone(
    newmodulelist = worker.newmodulelist.value(), # taken from worker configuration, need to stay in synch
    dqmDir = worker.folder.value(), # taken from worker configuration, need to stay in synch
)
