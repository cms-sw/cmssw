import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

ppsAlignmentHarvester = DQMEDHarvester("PPSAlignmentHarvester",
	folder = cms.string("CalibPPS/Common"),
	debug = cms.bool(True)
)