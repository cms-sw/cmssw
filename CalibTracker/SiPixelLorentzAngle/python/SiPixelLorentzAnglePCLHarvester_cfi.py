import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from CalibTracker.SiPixelLorentzAngle.SiPixelLorentzAnglePCLWorker_cfi import SiPixelLorentzAnglePCLWorker as worker

SiPixelLorentzAnglePCLHarvester = DQMEDHarvester(
    "SiPixelLorentzAnglePCLHarvester",
    newmodulelist = cms.vstring(worker.newmodulelist.value()), # taken from worker configuration, need to stay in synch
    dqmDir = cms.string(worker.folder.value()), # taken from worker configuration, need to stay in synch
    record = cms.string("SiPixelLorentzAngleRcd"),
    fitProbCut = cms.double(0.5)
)
