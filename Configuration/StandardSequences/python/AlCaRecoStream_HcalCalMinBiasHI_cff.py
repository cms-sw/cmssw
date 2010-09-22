import FWCore.ParameterSet.Config as cms

# last update: $Date: 2009/03/26 12:03:01 $ by $Author: argiro $

# HCAL calibration with min.bias
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff import *

from DQMOffline.Configuration.AlCaRecoDQMHI_cff import *

pathALCARECOHcalCalMinBias = cms.Path(seqALCARECOHcalCalMinBias*ALCARECOHcalCalPhisymDQM)

from Configuration.EventContent.AlCaRecoOutput_cff import *

ALCARECOStreamHcalCalMinBias = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalMinBias',
	paths  = (pathALCARECOHcalCalMinBias),
	content = OutALCARECOHcalCalMinBias.outputCommands,
	selectEvents = OutALCARECOHcalCalMinBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)
