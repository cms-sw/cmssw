import FWCore.ParameterSet.Config as cms

# last update: $Date: 2008/08/25 11:36:00 $ by $Author: futyand $

# HCAL calibration with min.bias
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff import *

from DQMOffline.Configuration.AlCaRecoDQM_cff import *

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
