import FWCore.ParameterSet.Config as cms
from DQMServices.Components.EDMtoMEConverter_cfi import EDMtoMEConverter
from Geometry.VeryForwardGeometry.geometryRPFromDB_cfi import *
from CalibPPS.TimingCalibration.PPSDiamondSampicTimingCalibrationPCLHarvester_cfi import *


DQMStore = cms.Service("DQMStore")

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
dqmEnvPPSTimingSampicCalibration = DQMEDHarvester('DQMHarvestingMetadata',
                                                  subSystemFolder=cms.untracked.string('AlCaReco/PPSDiamondSampicTimingCalibrationPCL/AlignedChannels'))
                   
EDMtoMEConvertPPSTimingSampicCalibration = EDMtoMEConverter.clone()
EDMtoMEConvertPPSTimingSampicCalibration.lumiInputTag = cms.InputTag("EDMtoMEConvertPPSTimingSampicCalibration", "MEtoEDMConverterLumi")
EDMtoMEConvertPPSTimingSampicCalibration.runInputTag = cms.InputTag("EDMtoMEConvertPPSTimingSampicCalibration", "MEtoEDMConverterRun")
           
ALCAHARVESTPPSDiamondSampicTimingCalibration = cms.Sequence(EDMtoMEConvertPPSTimingSampicCalibration +
                                                            PPSDiamondSampicTimingCalibrationPCLHarvester +
                                                            dqmEnvPPSTimingSampicCalibration)
