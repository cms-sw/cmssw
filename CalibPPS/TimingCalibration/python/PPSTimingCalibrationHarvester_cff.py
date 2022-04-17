import FWCore.ParameterSet.Config as cms

from Geometry.VeryForwardGeometry.geometryRPFromDB_cfi import *
from CalibPPS.TimingCalibration.ppsTimingCalibrationPCLHarvester_cfi import *
from DQMServices.Components.EDMtoMEConverter_cfi import EDMtoMEConverter

EDMtoMEConvertPPSTimingCalibration = EDMtoMEConverter.clone()
EDMtoMEConvertPPSTimingCalibration.lumiInputTag = cms.InputTag("MEtoEDMConvertPPSTimingCalib", "MEtoEDMConverterLumi")
EDMtoMEConvertPPSTimingCalibration.runInputTag = cms.InputTag("MEtoEDMConvertPPSTimingCalib", "MEtoEDMConverterRun")

                                        
ALCAHARVESTPPSTimingCalibration = cms.Task(
    EDMtoMEConvertPPSTimingCalibration,
    ppsTimingCalibrationPCLHarvester
)
