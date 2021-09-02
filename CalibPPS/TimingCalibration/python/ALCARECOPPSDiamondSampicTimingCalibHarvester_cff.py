import FWCore.ParameterSet.Config as cms

from Geometry.VeryForwardGeometry.geometryRPFromDB_cfi import *
from CalibPPS.TimingCalibration.PPSDiamondSampicTimingCalibrationPCLHarvester_cfi import *


DQMStore = cms.Service("DQMStore")

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
dqmEnv = DQMEDHarvester('DQMHarvestingMetadata',
                              subSystemFolder=cms.untracked.string('AlCaReco/PPSDiamondSampicTimingCalibrationPCL/AlignedChannels'))
                              
ALCAHARVESTPPSDiamondSampicTimingCalibration = cms.Sequence(PPSDiamondSampicTimingCalibrationPCLHarvester + dqmEnv)
