import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dtDAQInfo = DQMEDHarvester("DTDAQInfo",
		checkUros  = cms.untracked.bool(False)
	)


